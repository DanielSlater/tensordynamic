from collections import namedtuple

import numpy as np

import tensorflow as tf

from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.utils import xavier_init, tf_resize
from tensor_dynamic.weight_functions import noise_weight_extender


class BaseLayer(object):
    OUTPUT_BOUND_VALUE = 'output'
    INPUT_BOUND_VALUE = 'input'
    _bound_variable = namedtuple('BoundVariable', ['dimensions', 'variable'])

    def __init__(self,
                 input_layer,
                 output_nodes,
                 session,
                 weight_extender_func=noise_weight_extender,
                 name=None,
                 freeze=False,
                 deterministic=True):
        if not isinstance(input_layer, BaseLayer):
            raise TypeError("input_layer must be of type %s" % BaseLayer)

        self._session = session
        self._name = name
        self._input_layer = input_layer
        self._output_nodes = output_nodes
        self._input_nodes = self._input_layer._output_nodes
        self._next_layer = None
        self._weight_extender_func = weight_extender_func
        self._freeze = freeze
        self._deterministic = deterministic
        self._bound_variables = []
        input_layer._attach_next_layer(self)

    def name_scope(self):
        name = str(self.layer_number) + "_" + self._name
        return tf.name_scope(name)

    @property
    def is_input_layer(self):
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
    def deterministic(self):
        return self._deterministic

    @property
    def can_toggle_deterministic(self):
        try:
            self.deterministic = self.deterministic
        except AttributeError:
            # easier to ask forgiveness, etc..
            return False

        return True

    @property
    def activation(self):
        raise NotImplementedError()

    @property
    def bactivation(self):
        raise NotImplementedError()

    @property
    def has_bactivation(self):
        return False

    @property
    def output_shape(self):
        return self.activation.get_shape().as_list()

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
            raise Exception("Can attach_next_layer to Layer: %s which already is attached" % self._name)
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
    def output_layer_activation(self):
        return self.last_layer.activation

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
    def all_connected_layers(self):
        for u in list(reversed(list(self.upstream_layers))):
            yield u
        yield self
        for d in self.downstream_layers:
            yield d

    def supervised_cost(self, targets):
        return None

    def unsupervised_cost(self):
        return None

    def cost(self, targets):
        supervised_cost = self.supervised_cost(targets)
        unsupervised_cost = self.unsupervised_cost()

        if supervised_cost:
            if unsupervised_cost:
                return supervised_cost + unsupervised_cost
            return supervised_cost
        if unsupervised_cost:
            return unsupervised_cost

        return None

    def cost_all_layers(self, targets):
        all_costs = [x.cost(targets) for x in self.all_connected_layers]
        all_costs = filter(lambda v: v is not None, all_costs)
        return tf.add_n(all_costs)

    def set_all_deterministic(self, value):
        for layer in self.all_connected_layers:
            if layer.can_toggle_deterministic:
                layer.deterministic = value

    @property
    def layer_number(self):
        return len(list(self.upstream_layers))

    @property
    def kargs(self):
        return {
            'weight_extender_func': self._weight_extender_func,
            'freeze': self._freeze,
            'deterministic': self._deterministic,
            'name': self._name}

    def resize_needed(self):
        if self._input_layer.output_shape[-1] != self.input_nodes:
            return True
        return False

    def resize(self, new_output_nodes=None):
        new_input_nodes = int(self.input_shape[-1])
        input_nodes_changed = new_input_nodes != self._input_nodes
        output_nodes_changed = new_output_nodes and new_output_nodes != self._output_nodes
        self._output_nodes = new_output_nodes
        self._input_nodes = new_input_nodes

        if input_nodes_changed and self.bactivate:
            tf_resize(self._session, self.bactivation, (None, self._input_nodes))

        if output_nodes_changed:
            tf_resize(self._session, self.activation, (None, self._output_nodes))

        for bound_variable in self._bound_variables:
            if input_nodes_changed and self._bound_dimensions_contains_input(bound_variable.dimensions) or \
                            output_nodes_changed and self._bound_dimensions_contains_output(bound_variable.dimensions):
                int_dims = self._bound_dimsions_to_ints(bound_variable.dimensions)
                tf_resize(self._session, bound_variable.variable, int_dims,
                          self._weight_extender_func(self._session.run(bound_variable.variable), int_dims))

        if self._next_layer and self._next_layer.resize_needed():
            self._next_layer.resize()

    def _bound_dimsions_to_ints(self, bound_dims):
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

    def _create_variable(self, bound_dimensions, default_val, name):
        int_dims = self._bound_dimsions_to_ints(bound_dimensions)

        with self.name_scope():
            if isinstance(default_val, np.ndarray):
                default_val = self._weight_extender_func(default_val, int_dims)

            var = tf.Variable(default_val, trainable=not self._freeze, name=name)

            self._session.run(tf.initialize_variables([var]))
            self._bound_variables.append(self._bound_variable(bound_dimensions, var))
            return var

    def _bound_dimensions_contains_input(self, bound_dimensions):
        return any(x for x in bound_dimensions if x == self.INPUT_BOUND_VALUE)

    def _bound_dimensions_contains_output(self, bound_dimensions):
        return any(x for x in bound_dimensions if x == self.OUTPUT_BOUND_VALUE)
