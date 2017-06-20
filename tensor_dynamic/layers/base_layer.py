from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensor_dynamic.lazyprop import clear_all_lazyprops, lazyprop
from tensor_dynamic.utils import tf_resize
from tensor_dynamic.weight_functions import noise_weight_extender, array_extend


class BaseLayer(object):
    __metaclass__ = ABCMeta

    OUTPUT_BOUND_VALUE = 'output'
    INPUT_BOUND_VALUE = 'input'
    INPUT_DIM_3_BOUND_VALUE = 'input_3'
    OUTPUT_DIM_3_BOUND_VALUE = 'output_3'

    _BoundVariable = namedtuple('_BoundVariable', ['name', 'dimensions', 'variable', 'is_kwarg'])

    def __init__(self,
                 input_layer,
                 output_nodes,
                 session=None,
                 weight_extender_func=noise_weight_extender,
                 name=None,
                 freeze=False):
        """Base class from which all layers will inherit. This is an abstract class

        Args:
            input_layer (tensor_dynamic.base_layer.BaseLayer): This layer will work on the activation of the input_layer
            output_nodes (int | tuple of ints): Number of output nodes for this layer, can be a tuple of multi dimensional output, e.g. convolutional network
            session (tensorflow.Session): The session within which all these variables should be created
            weight_extender_func (func): Method that extends the size of matrix or vectors
            name (str): Used for identifying the layer and when initializing tensorflow variables
            freeze (bool):If True then weights in this layer are not trainable
        """
        if not isinstance(input_layer, BaseLayer):
            raise TypeError("input_layer must be of type %s" % BaseLayer)

        assert isinstance(output_nodes, (int, tuple))
        assert isinstance(input_layer, BaseLayer)

        self._session = session or input_layer.session
        self._name = name
        self._input_layer = input_layer
        self._output_nodes = (output_nodes,) if type(output_nodes) == int else output_nodes
        self._input_nodes = self._input_layer._output_nodes
        self._next_layer = None
        self._weight_extender_func = weight_extender_func
        self._freeze = freeze
        self._bound_variables = []
        input_layer._attach_next_layer(self)

    def name_scope(self):
        """Used for naming variables associated with this layer in TensorFlow in a consistent way

        Format = "{layer_number}_{layer_name}"

        Examples:
            with self.name_scope():
                my_new_variable = tf.Variable(default_val, name="name")

        Returns:
            A context manager that installs `name` as a new name scope in the
            default graph.
        """
        name = str(self.layer_number) + "_" + self._name
        return tf.name_scope(name)

    @lazyprop
    def activation_train(self):
        """The activation used for training this layer, this will often be the same as prediction except with dropout or
        random noise applied.

        Returns:
            tensorflow.Tensor
        """
        return self._layer_activation(self.input_layer.activation_train, True)

    @lazyprop
    def activation_predict(self):
        """The activation used for predictions from this layer, this will often be the same as training except without
        dropout or random noise applied.

        Returns:
            tensorflow.Tensor
        """
        return self._layer_activation(self.input_layer.activation_predict, False)

    @abstractmethod
    def _layer_activation(self, input_tensor, is_train):
        """The activation for this layer

        Args:
            input_tensor (tensorflow.Tensor):
            is_train (bool): If true this is activation for training, if false for prediction

        Returns:
            tensorflow.Tensor
        """
        raise NotImplementedError()

    @property
    def bactivate(self):
        """All layers have output activation, some unsupervised layer activate backwards as well.

        e.g. Layers in ladder networks, Denoising Autoencoders

        Returns:
            bool
        """
        return False

    @property
    def is_input_layer(self):
        """Are we the input layer

        Returns:
            bool
        """
        return False

    @property
    def is_output_layer(self):
        """Is this the final layer of the network

        Returns:
            bool
        """
        return self._next_layer is None

    @property
    def output_nodes(self):
        """The number of output nodes

        Returns:
            tuple of ints
        """
        return self._output_nodes

    @property
    def input_nodes(self):
        """The number of input nodes to this layer

        Returns:
            tuple of ints
        """
        return self._input_nodes

    @property
    def session(self):
        """Session used to create the variables in this layer

        Returns:
            tensorflow.Session
        """
        return self._session

    @lazyprop
    def bactivation_train(self):
        """The activation used for training this layer, this will often be the same as prediction except with dropout or
        random noise applied.

        Returns:
            tensorflow.Tensor
        """
        return self._layer_bactivation(self.activation_train, True)

    @lazyprop
    def bactivation_predict(self):
        """The activation used for predictions from this layer, this will often be the same as training except without
        dropout or random noise applied.

        Returns:
            tensorflow.Tensor
        """
        return self._layer_bactivation(self.activation_predict, False)

    # @abstractmethod
    def _layer_bactivation(self, input_tensor, is_train):
        """The bactivation for this layer

        Args:
            input_tensor (tensorflow.Tensor):
            is_train (bool): If true this is activation for training, if false for prediction

        Returns:
            tensorflow.Tensor
        """
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
    def target_placeholder(self):
        return self.last_layer.target_placeholder

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

    def activate_predict(self, data_set):
        """Get the prediction activation of this network given the data_set as input

        Args:
            data_set (np.array): np.array or Array matching the dimensions of the input placeholder

        Returns:
            np.array: prediction activation of the network
        """
        return self.session.run(self.activation_predict, feed_dict={self.input_placeholder: data_set})

    def supervised_cost_train(self, targets):
        return None

    def unsupervised_cost_train(self):
        return None

    def cost_train(self, targets):
        supervised_cost = self.supervised_cost_train(targets)
        unsupervised_cost = self.unsupervised_cost_train()

        if supervised_cost is not None:
            if unsupervised_cost is not None:
                return supervised_cost + unsupervised_cost
            return supervised_cost
        if unsupervised_cost is not None:
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
            if bound_variable.is_kwarg:
                kwarg_dict[bound_variable.name] = self.session.run(bound_variable.variable)

        return kwarg_dict

    def clone(self, session=None):
        """Produce a clone of this layer AND all connected upstream layers

        Args:
            session (tensorflow.Session): If passed in the clone will be created with all variables initialised in this session
                                          If None then the current session of this layer is used

        Returns:
            tensorflow_dynamic.BaseLayer: A copy of this layer and all upstream layers
        """
        new_self = self.__class__(self.input_layer.clone(session or self.session),
                                  self.output_nodes,
                                  session=session or self._session,
                                  **self.kwargs)

        return new_self

    def resize_needed(self):
        """ If there is a mismatch between the input size of this layer and the output size of it's previous layer will
        return True

        Returns:
            bool
        """
        if self._input_layer.output_nodes != self.input_nodes:
            return True
        return False

    def resize(self, new_output_nodes=None,
               output_nodes_to_prune=None,
               input_nodes_to_prune=None,
               split_output_nodes=None,
               split_input_nodes=None,
               split_nodes_noise_std=.1):
        """Resize this layer by changing the number of output nodes. Will also resize any downstream layers

        Args:
            new_output_nodes (int | tuple of ints): If passed we change the number of output nodes of this layer to be new_output_nodes
            output_nodes_to_prune ([int]): list of indexes of the output nodes we want pruned e.g. [1, 3] would remove
                the 1st and 3rd output node from this layer
            input_nodes_to_prune ([int]): list of indexes of the input nodes we want pruned e.g. [1, 3] would remove the
                1st and 3rd input node from this layer
            split_output_nodes ([int]): list of indexes of nodes to split. This is for growing the layer
            split_input_nodes: ([int]): list of indexes of nodes that where split in the previous layer.
            split_nodes_noise_std (float): standard deviation of noise to add when splitting a node
        """
        if isinstance(new_output_nodes, int):
            new_output_nodes = (new_output_nodes,)
        elif new_output_nodes is not None and not isinstance(new_output_nodes, tuple):
            raise ValueError("new_output_nodes must be tuple of int %s" % (new_output_nodes,))

        if output_nodes_to_prune:
            if split_output_nodes:
                raise NotImplementedError("At the moment must either split or prune")
            if not (new_output_nodes is None or new_output_nodes != self._output_nodes - len(
                    output_nodes_to_prune)):  # TODO, needs some work
                raise Exception("Different number of output nodes set from that left after pruning")
            else:
                new_output_nodes = self._output_nodes - len(output_nodes_to_prune)
        elif split_output_nodes:
            if not (new_output_nodes is None or new_output_nodes != self._output_nodes + len(
                    split_output_nodes)):  # TODO, needs some work
                raise Exception("Different number of output nodes set from that left after splitting")
            else:
                new_output_nodes = self._output_nodes + len(split_output_nodes)
        else:
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

                if isinstance(bound_variable.variable, tf.Variable):
                    old_values = self._session.run(bound_variable.variable)
                    if output_nodes_to_prune or split_output_nodes:
                        output_bound_axis = bound_variable.dimensions.index(self.OUTPUT_BOUND_VALUE)
                        if output_nodes_to_prune:
                            old_values = np.delete(old_values, output_nodes_to_prune, output_bound_axis)
                        else:  # split
                            old_values = array_extend(old_values, {output_bound_axis: split_output_nodes},
                                                      noise_std=split_nodes_noise_std)
                    if input_nodes_to_prune or split_input_nodes:
                        input_bound_axis = bound_variable.dimensions.index(self.INPUT_BOUND_VALUE)
                        if input_nodes_to_prune:
                            old_values = np.delete(old_values, input_nodes_to_prune, input_bound_axis)
                        else:  # split
                            old_values = array_extend(old_values, {output_bound_axis: split_output_nodes},
                                                      noise_std=split_nodes_noise_std)

                    new_values = self._weight_extender_func(old_values, int_dims)

                    tf_resize(self._session, bound_variable.variable, int_dims,
                              new_values)
                else:
                    # this is a tensor, not a variable so has no weights
                    tf_resize(self._session, bound_variable.variable, int_dims)

        if output_nodes_changed:
            tf_resize(self._session, self.activation_train, (None,) + self._output_nodes)
            tf_resize(self._session, self.activation_predict, (None,) + self._output_nodes)

        if input_nodes_changed and self.bactivate:
            tf_resize(self._session, self.bactivation_train, (None,) + self._input_nodes)
            tf_resize(self._session, self.bactivation_predict, (None,) + self._input_nodes)

        if self._next_layer and self._next_layer.resize_needed():
            self._next_layer.resize(input_nodes_to_prune=output_nodes_to_prune, split_input_nodes=split_output_nodes)

    def _bound_dimensions_to_ints(self, bound_dims):
        int_dims = ()
        for x in bound_dims:
            if isinstance(x, int):
                if x == -1:
                    int_dims += (None,)
                else:
                    int_dims += (x,)
            elif x == self.OUTPUT_BOUND_VALUE:
                int_dims += (self._output_nodes[0], )
            elif x == self.INPUT_BOUND_VALUE:
                int_dims += (self._input_nodes[0], )
            elif x == self.INPUT_DIM_3_BOUND_VALUE:
                assert len(self._input_nodes) == 3, "must have 3 input dimensions"
                int_dims += (self._input_nodes[2], )
            elif x == self.OUTPUT_DIM_3_BOUND_VALUE:
                assert len(self._input_nodes) == 3, "must have 3 output dimensions"
                int_dims += (self._output_nodes[2], )
            else:
                raise Exception("bound dimension must be either int or 'input' or 'output' found %s" % (x,))
        return int_dims

    def _create_variable(self, name, bound_dimensions, default_val, is_kwarg=True, is_trainable=True):
        int_dims = self._bound_dimensions_to_ints(bound_dimensions)

        with self.name_scope():
            if isinstance(default_val, np.ndarray):
                default_val = self._weight_extender_func(default_val, int_dims)

            var = tf.Variable(default_val, trainable=(not self._freeze) and is_trainable, name=name)

            self._session.run(tf.initialize_variables([var]))
            self._bound_variables.append(self._BoundVariable(name, bound_dimensions, var, is_kwarg))
            return var

    def _register_variable(self, name, bound_dimensions, variable, is_constructor_variable=True):
        """Register a variable that will need to be resized with this layer

        Args:
            name (str): Name used for displaying errors and debuging
            bound_dimensions (tuple of (self.OUTPUT_BOUND_VALUE or self.INPUT_BOUND_VALUE or int)):
            variable (tf.Tensor): The variable to bind
            is_constructor_variable (bool): If true this variable is passed as an arg to the constructor of this class if it is cloned
        """
        int_dims = self._bound_dimensions_to_ints(bound_dimensions)
        assert tuple(variable.get_shape().as_list()) == tuple(int_dims)
        self._bound_variables.append(self._BoundVariable(name, bound_dimensions, variable, is_constructor_variable))

    def _bound_dimensions_contains_input(self, bound_dimensions):
        return any(x for x in bound_dimensions if x == self.INPUT_BOUND_VALUE or x == self.INPUT_DIM_3_BOUND_VALUE)

    def _bound_dimensions_contains_output(self, bound_dimensions):
        return any(x for x in bound_dimensions if x == self.OUTPUT_BOUND_VALUE or x == self.OUTPUT_DIM_3_BOUND_VALUE)

    def detach_output(self):
        """Detaches the connect between this layer and the next layer

        Returns:
            BaseLayer : The next layer, now detached from this layer
        """
        if self._next_layer is None:
            raise ValueError("Cannot detach_output if there is no next layer")
        next_layer = self._next_layer

        next_layer._input_layer = None
        clear_all_lazyprops(next_layer)

        self._next_layer = None
        clear_all_lazyprops(self)

        return next_layer

    def add_intermediate_layer(self, layer_creation_func):
        """Adds a layer to the network between this layer

        Args:
            layer_creation_func (BaseLayer->None): Method that creates the intermediate layer, takes this layer as a
                parameter
        """
        old_next_layer = self.detach_output()
        new_next_layer = layer_creation_func(self)

        new_next_layer._next_layer = old_next_layer
        old_next_layer._input_layer = new_next_layer

    @property
    def assign_op(self):
        """Optional tensor flow op that will be set as a dependency of the train step. Useful for things like clamping
        variables, or setting mean/var in batch normalization layer

        Returns:
            tensorflow.Operation or None
        """
        return None

    @property
    def variables(self):
        """Get all the tensorflow variables used in this layer, useful for weight regularization

        Returns:
            Iterable of tf.Variable:
        """
        for bound_variable in self._bound_variables:
            yield bound_variable.variable

    @property
    def variables_all_layers(self):
        """Get all the tensorflow variables used in all connected layers, useful for weight regularization

        Returns:
            Iterable of tf.Variable:
        """
        for layer in self.all_layers:
            for variable in layer.variables:
                yield variable