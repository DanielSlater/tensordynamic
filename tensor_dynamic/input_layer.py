import tensorflow as tf
from tensor_dynamic.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop
from tensorflow.python import control_flow_ops


class InputLayer(BaseLayer):
    def __init__(self, placeholder, name='Input'):
        if isinstance(placeholder, tuple):
            placeholder = tf.placeholder('float', placeholder)
        elif isinstance(placeholder, int):
            placeholder = tf.placeholder('float', (None, placeholder))
        self._name = name
        self._placeholder = placeholder
        self._next_layer = None
        self._input_layer = None
        self._deterministic = True

    @property
    def activation(self):
        return self._placeholder

    @property
    def first_layer(self):
        return self

    @property
    def bactivation(self):
        if not self._next_layer:
            raise Exception("Cannot get bactivation with no attached layer")
        return self._next_layer.bactivation

    @property
    def input_shape(self):
        raise Exception("Input layer has no input shape")

    @property
    def input_placeholder(self):
        return self._placeholder


class NoisyInputLayer(InputLayer):
    def __init__(self, placeholder, session, noise_std=1.0, name='NoisyInput'):
        super(NoisyInputLayer, self).__init__(placeholder, name)
        self._noise_std_value = noise_std
        self._session = session
        with tf.name_scope(name):
            self._noise_std = tf.Variable(noise_std, name='noise_std')
        self._session.run(tf.initialize_variables([self._noise_std]))

    @property
    def deterministic(self):
        return self._session.run(self._noise_std) < 0.00001

    @deterministic.setter
    def deterministic(self, value):
        if value:
            self._session.run(tf.assign(self._noise_std, 0.0))
        else:
            self._session.run(tf.assign(self._noise_std, self._noise_std_value))

    @lazyprop
    def activation_noisy(self):
        return self._placeholder + tf.random_normal(tf.shape(self._placeholder),
                                                    stddev=self._noise_std)

    @property
    def z(self):
        return self.activation_noisy

    @lazyprop
    def activation(self):
        return self._placeholder