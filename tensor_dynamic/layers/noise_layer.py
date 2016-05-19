import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop


class NoiseLayer(BaseLayer):
    def __init__(self, input_layer, session, noise_std=1.0, name='NoiseLayer'):
        super(NoiseLayer, self).__init__(input_layer, input_layer.output_nodes, session, name=name)
        self._noise_std = noise_std
        self._session = session

    @lazyprop
    def activation_train(self):
        return self.input_layer.activation_train + tf.random_normal(tf.shape(self.input_layer.activation_train),
                                                                    stddev=self._noise_std)

    @property
    def activation_predict(self):
        return self.input_layer.activation_predict
