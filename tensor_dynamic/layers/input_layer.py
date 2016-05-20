import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop


class InputLayer(BaseLayer):
    def __init__(self, placeholder, name='Input'):
        """

        Parameters
        ----------
        placeholder(tensorflow.placeholder
        name(str): the name for this layer

        Returns
        -------

        """
        if isinstance(placeholder, tuple):
            placeholder = tf.placeholder('float', placeholder)
        elif isinstance(placeholder, int):
            placeholder = tf.placeholder('float', (None, placeholder))

        self._output_nodes = int(placeholder.get_shape()[-1])
        self._name = name
        self._placeholder = placeholder
        self._next_layer = None
        self._input_layer = None
        self._deterministic = True

    @property
    def activation(self):
        return self._placeholder

    @property
    def activation_train(self):
        return self._placeholder

    @property
    def activation_predict(self):
        return self._placeholder

    @property
    def first_layer(self):
        return self

    @property
    def bactivate(self):
        return False

    @property
    def input_shape(self):
        raise Exception("Input layer has no input shape")

    @property
    def input_placeholder(self):
        return self._placeholder

    @property
    def is_input_layer(self):
        return True

    def clone(self, session):
        return self.__class__(self._placeholder, name=self._name)


class NoisyInputLayer(InputLayer):
    def __init__(self, placeholder, session, noise_std=1.0, name='NoisyInput'):
        super(NoisyInputLayer, self).__init__(placeholder, name)
        self._noise_std = noise_std
        self._session = session
        with self.name_scope():
            self._predict = tf.Variable(noise_std, name='predict')
            self._session.run(tf.initialize_variables([self._predict]))

    @lazyprop
    def activation_train(self):
        return self.activation + tf.random_normal(tf.shape(self._placeholder),
                                                  stddev=self._noise_std)

    @property
    def activation_predict(self):
        return self.activation

    def clone(self, session):
        return self.__class__(self._placeholder, session, noise_std=self._noise_std, name=self._name)
