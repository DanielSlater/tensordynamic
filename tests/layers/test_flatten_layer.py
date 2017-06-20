import numpy as np
import tensorflow as tf

from tensor_dynamic.layers.convolutional_layer import ConvolutionalLayer
from tensor_dynamic.layers.flatten_layer import FlattenLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tests.layers.base_layer_testcase import BaseLayerWrapper


class TestFlattenLayer(BaseLayerWrapper.BaseLayerTestCase):
    INPUT_NODES = (10, 10, 4)

    def _create_layer_for_test(self):
        return FlattenLayer(self._input_layer, session=self.session)

    def test_create_layer(self):
        input_p = tf.placeholder("float", (None, 10, 10, 4))
        layer = FlattenLayer(InputLayer(input_p), session=self.session)

        self.assertEqual(layer.activation_predict.get_shape().as_list(), [None, 10 * 10 * 4])

    def test_reshape(self):
        input_vals = np.random.normal(size=(1, 2, 2, 1)).astype(np.float32)

        convolution_nodes = (2, 2, 2)
        input_p = tf.placeholder("float", (None, 2, 2, 1))
        layer = ConvolutionalLayer(InputLayer(input_p), convolution_nodes, session=self.session)
        flatten = FlattenLayer(layer, session=self.session)
        result1 = self.session.run(layer.activation_predict, feed_dict={flatten.input_placeholder: input_vals})

        layer.resize(3)
        result2 = self.session.run(layer.activation_predict, feed_dict={flatten.input_placeholder: input_vals})

        print(result1)
        print(result2)

        self.assertEquals(result2.shape[3], 3)

    def test_resize(self):
        convolution_nodes = (2, 2, 2)
        input_p = tf.placeholder("float", (None, 2, 2, 1))
        layer = ConvolutionalLayer(InputLayer(input_p), convolution_nodes, session=self.session)
        flatten = FlattenLayer(layer, session=self.session)

        self.assertEqual(flatten.output_nodes[0], 2 * 2 * 2)

        layer.resize(3)

        self.assertEqual(flatten.output_nodes[0], 2 * 2 * 3)
