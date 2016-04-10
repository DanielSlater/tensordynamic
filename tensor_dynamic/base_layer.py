import numpy as np

import tensorflow as tf

from tensor_dynamic.utils import xavier_init
from tensor_dynamic.weight_functions import noise_weight_extender


class BaseLayer(object):
    def __init__(self, input_layer,
                 weight_extender_func=noise_weight_extender,
                 name=None,
                 freeze=False,
                 deterministic=True):
        if not isinstance(input_layer, BaseLayer):
            raise TypeError("input_layer must be of type %s" % BaseLayer)

        self._name = name
        self._input_layer = input_layer
        self._next_layer = None
        self._weight_extender_func = weight_extender_func
        self._freeze = freeze
        self._deterministic = deterministic
        self._variables = []
        input_layer._attach_next_layer(self)

    def initialize_variables(self, session):
        session.run(tf.initialize_variables(self._variables))

    def _create_variable(self, dims, default_val, name):
        with tf.name_scope(self._name):
            if isinstance(default_val, np.ndarray):
                default_val = self._weight_extender_func(default_val, dims)

            var = tf.Variable(default_val, trainable=not self._freeze, name=name)
            self._variables.append(var)
            return var

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