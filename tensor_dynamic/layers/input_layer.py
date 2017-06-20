import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop


class InputLayer(BaseLayer):
    def __init__(self, placeholder, name='Input'):
        """Input layer to a neural network

        Args:
            placeholder (tensorflow.placeholder or (int) or int): If an int then a tensorflow.placeholder is created
                with dimensions (None, placholder) if a tuple a placeholder is created of that dimension
            name(str): the name for this layer
        """
        if isinstance(placeholder, (tuple, list)):
            placeholder = tf.placeholder('float', placeholder)
        elif isinstance(placeholder, int):
            placeholder = tf.placeholder('float', (None, placeholder))

        self._output_nodes = tuple(int(x) for x in placeholder.get_shape()[1:])
        self._name = name
        self._placeholder = placeholder
        self._next_layer = None
        self._input_layer = None

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

    def clone(self, session=None):
        return self.__class__(self._placeholder, name=self._name)

    @property
    def variables(self):
        return ()

    def _layer_activation(self, _1, _2):
        pass


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


class SemiSupervisedInputLayer(InputLayer):
    def __init__(self, input_dim, name='Input'):
        if isinstance(input_dim, tuple):
            supervised = tf.placeholder('float', input_dim)
            unsupervised = tf.placeholder('float', input_dim)
        elif isinstance(input_dim, int):
            supervised = tf.placeholder('float', (None, input_dim))
            unsupervised = tf.placeholder('float', (None, input_dim))

        super(SemiSupervisedInputLayer, self).__init__(supervised, name=name)
        self._unsupervised_placeholder = unsupervised

    @property
    def unsupervised_placeholder(self):
        return self._unsupervised_placeholder

    @lazyprop
    def labeled_input_size(self):
        return tf.shape(self._placeholder)[1]