from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensor_dynamic.utils import tf_resize
from tensor_dynamic.weight_functions import noise_weight_extender


class BaseLayer(object):
    OUTPUT_BOUND_VALUE = 'output'
    INPUT_BOUND_VALUE = 'input'
    _BoundVariable = namedtuple('BoundVariable', ['name', 'dimensions', 'variable'])

    def __init__(self,
                 input_layer,
                 output_nodes,
                 session=None,
                 weight_extender_func=noise_weight_extender,
                 name=None,
                 freeze=False):
        """
        Base class from which all layers will inherit. This is an abstract class

        Parameters
        ----------
        input_layer : tensor_dynamic.base_layer.BaseLayer
            This layer will work on the activation of the input_layer
        output_nodes: int
            Number of output nodes for this layer
        session : tensorflow.Session
            The session within which all these variables should be created
        weight_extender_func: func
            Method that extends the size of matrix or vectors
        name : str
            Used for identifying the layer and when initializing tensorflow variables
        freeze : bool
            If True then weights in this layer are not trainable
        """
        if not isinstance(input_layer, BaseLayer):
            raise TypeError("input_layer must be of type %s" % BaseLayer)

        self._session = session or input_layer.session
        self._name = name
        self._input_layer = input_layer
        self._output_nodes = output_nodes
        self._input_nodes = self._input_layer._output_nodes
        self._next_layer = None
        self._weight_extender_func = weight_extender_func
        self._freeze = freeze
        self._bound_variables = []
        input_layer._attach_next_layer(self)

    def name_scope(self):
        """
        Used for naming variables associated with this layer in TensorFlow in a consistent way

        Format = "{layer_number}_{layer_name}"

        Returns
        -------
        A context manager that installs `name` as a new name scope in the
        default graph.
        """
        name = str(self.layer_number) + "_" + self._name
        return tf.name_scope(name)

    @property
    def activation_train(self):
        """
        The activation used for training this layer, this will often be the same as prediction except with dropout or
        random noise applied.

        Returns
        -------
        tensorflow.Tensor
        """
        raise NotImplementedError()

    @property
    def activation_predict(self):
        raise NotImplementedError()

    @property
    def bactivate(self):
        """
        All layers have output activation, some unsupervised layer activate backwards as well.

        e.g. Layers in ladder networks, Denoising Autoencoders

        Returns
        -------
        bool
        """
        return False

    @property
    def is_input_layer(self):
        """
        Are we the input layer

        Returns
        -------
        bool
        """
        return False

    @property
    def is_output_layer(self):
        return self._next_layer is None

    @property
    def output_nodes(self):
        return self._output_nodes

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def session(self):
        return self._session

    @property
    def bactivation_predict(self):
        raise NotImplementedError()

    @property
    def bactivation_train(self):
        raise NotImplementedError()

    @property
    def has_bactivation(self):
        return False

    @property
    def output_shape(self):
        return self.activation_predict.get_shape().as_list()

    @property
    def input_shape(self):
        return self._input_layer.output_shape

    @property
    def next_layer(self):
        return self._next_layer

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def has_next_layer(self):
        return self.next_layer

    def _attach_next_layer(self, layer):
        if self.has_next_layer:
            raise Exception("Can not attach_next_layer to Layer: %s which already has a next layer" % self._name)
        if not isinstance(layer, BaseLayer):
            raise TypeError("Attached layer must be of type %s" % BaseLayer)

        self._next_layer = layer

    @property
    def last_layer(self):
        if self._next_layer is not None:
            return self._next_layer.last_layer
        return self

    @property
    def first_layer(self):
        return self.input_layer.first_layer

    @property
    def input_placeholder(self):
        return self.first_layer.input_placeholder

    @property
    def downstream_layers(self):
        if self._next_layer:
            yield self._next_layer
            for d in self._next_layer.downstream_layers:
                yield d

    @property
    def upstream_layers(self):
        if self._input_layer:
            yield self._input_layer
            for d in self._input_layer.upstream_layers:
                yield d

    @property
    def all_layers(self):
        return self.all_connected_layers

    @property
    def all_connected_layers(self):
        for u in list(reversed(list(self.upstream_layers))):
            yield u
        yield self
        for d in self.downstream_layers:
            yield d

    def supervised_cost_train(self, targets):
        return None

    def unsupervised_cost_train(self):
        return None

    def cost_train(self, targets):
        supervised_cost = self.supervised_cost_train(targets)
        unsupervised_cost = self.unsupervised_cost_train()

        if supervised_cost:
            if unsupervised_cost:
                return supervised_cost + unsupervised_cost
            return supervised_cost
        if unsupervised_cost:
            return unsupervised_cost

        return None

    def cost_all_layers_train(self, targets):
        all_costs = [x.cost_train(targets) for x in self.all_connected_layers]
        all_costs = filter(lambda v: v is not None, all_costs)
        return tf.add_n(all_costs, name="cost_all_layers")

    @property
    def layer_number(self):
        return len(list(self.upstream_layers))

    @property
    def kwargs(self):
        kwargs = {
            'weight_extender_func': self._weight_extender_func,
            'freeze': self._freeze,
            'name': self._name}
        kwargs.update(self._bound_variables_as_kwargs())
        return kwargs

    def _bound_variables_as_kwargs(self):
        kwarg_dict = {}
        for bound_variable in self._bound_variables:
            kwarg_dict[bound_variable.name] = self.session.run(bound_variable.variable)

        return kwarg_dict

    def clone(self, session=None):
        """
        Produce a clone of this layer AND all connected upstream layers

        Parameters
        ----------
        session : tensorflow.Session
            If passed in the clone will be created with all variables initialised in this session
            If None then the current session of this layer is used

        Returns
        -------
        tensorflow_dynamic.BaseLayer

        A copy of this layer and all upstream layers
        """
        new_self = self.__class__(self.input_layer.clone(session or self.session),
                                  self.output_nodes,
                                  session=session or self._session,
                                  **self.kwargs)

        return new_self

    def resize_needed(self):
        """
        If there is a mismatch between the input size of this layer and the output size of it's previous layer will
        return True

        Returns
        -------
        bool
        """
        if self._input_layer.output_nodes != self.input_nodes:
            return True
        return False

    def resize(self, new_output_nodes=None):
        """
        Resize this layer by changing the number of output nodes. Will also resize any downstream layers

        Parameters
        ----------
        new_output_nodes : int
            If passed we change the number of output nodes of this layer to be new_output_nodes
            Otherwise we change the size to current output nodes+1
        """
        new_output_nodes = new_output_nodes or self._output_nodes
        new_input_nodes = self.input_layer.output_nodes
        input_nodes_changed = new_input_nodes != self._input_nodes
        output_nodes_changed = new_output_nodes != self._output_nodes

        self._output_nodes = new_output_nodes
        self._input_nodes = new_input_nodes

        for bound_variable in self._bound_variables:
            if input_nodes_changed and self._bound_dimensions_contains_input(bound_variable.dimensions) or \
                            output_nodes_changed and self._bound_dimensions_contains_output(bound_variable.dimensions):
                int_dims = self._bound_dimensions_to_ints(bound_variable.dimensions)
                tf_resize(self._session, bound_variable.variable, int_dims,
                          self._weight_extender_func(self._session.run(bound_variable.variable), int_dims))

        if output_nodes_changed:
            tf_resize(self._session, self.activation_train, (None, self._output_nodes))
            tf_resize(self._session, self.activation_predict, (None, self._output_nodes))

        if input_nodes_changed and self.bactivate:
            tf_resize(self._session, self.bactivation_train, (None, self._input_nodes))
            tf_resize(self._session, self.bactivation_predict, (None, self._input_nodes))

        if self._next_layer and self._next_layer.resize_needed():
            self._next_layer.resize()

    def _bound_dimensions_to_ints(self, bound_dims):
        int_dims = ()
        for x in bound_dims:
            if isinstance(x, int):
                int_dims += (x,)
            elif x == self.OUTPUT_BOUND_VALUE:
                int_dims += (self._output_nodes,)
            elif x == self.INPUT_BOUND_VALUE:
                int_dims += (self._input_nodes,)
            else:
                raise Exception("bound dimension must be either int or 'input' or 'output' found %s" % (x,))
        return int_dims

    def _create_variable(self, name, bound_dimensions, default_val):
        int_dims = self._bound_dimensions_to_ints(bound_dimensions)

        with self.name_scope():
            if isinstance(default_val, np.ndarray):
                default_val = self._weight_extender_func(default_val, int_dims)

            var = tf.Variable(default_val, trainable=not self._freeze, name=name)

            self._session.run(tf.initialize_variables([var]))
            self._bound_variables.append(self._BoundVariable(name, bound_dimensions, var))
            return var

    def _register_variable(self, name, bound_dimensions, variable):
        int_dims = self._bound_dimensions_to_ints(bound_dimensions)
        assert tuple(variable.get_shape().as_list()) == tuple(int_dims)
        self._bound_variables.append(self._BoundVariable(name, bound_dimensions, variable))

    def _bound_dimensions_contains_input(self, bound_dimensions):
        return any(x for x in bound_dimensions if x == self.INPUT_BOUND_VALUE)

    def _bound_dimensions_contains_output(self, bound_dimensions):
        return any(x for x in bound_dimensions if x == self.OUTPUT_BOUND_VALUE)
