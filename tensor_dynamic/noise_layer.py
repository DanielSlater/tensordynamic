import tensorflow as tf

from tensor_dynamic.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop


class NoiseLayer(BaseLayer):
    def __init__(self, input_layer, session, noise_std=1.0, name='NoiseLayer'):
        super(NoiseLayer, self).__init__(input_layer, input_layer.output_nodes, session, name=name)
        self._noise_std_value = noise_std
        self._session = session
        with self.name_scope():
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
    def activation(self):
        return self.input_layer.activation + tf.random_normal(tf.shape(self.input_layer.activation),
                                                    stddev=self._noise_std)