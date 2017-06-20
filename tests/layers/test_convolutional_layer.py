import numpy as np
import tensorflow as tf

from tensor_dynamic.layers.convolutional_layer import ConvolutionalLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tests.layers.base_layer_testcase import BaseLayerWrapper


class TestConvolutionalLayer(BaseLayerWrapper.BaseLayerTestCase):
    INPUT_NODES = (10, 10, 1)
    OUTPUT_NODES = (3, 3, 16)

    def _create_layer_for_test(self):
        return ConvolutionalLayer(self._input_layer, self.OUTPUT_NODES, session=self.session)

    def test_create_layer(self):
        convolution_dimensions = (5, 5, 4)
        input_p = tf.placeholder("float", (None, 10, 10, 1))
        layer = ConvolutionalLayer(InputLayer(input_p), convolution_dimensions, session=self.session)

        self.assertEqual(layer.activation_predict.get_shape().as_list(), [None, 10, 10, 4])

    def test_create_extra_weight_dimensions(self):
        convolutional_nodes = 3, 3, 16
        input_p = tf.placeholder("float", (None, 10, 10, 2))
        layer = ConvolutionalLayer(InputLayer(input_p), convolutional_nodes, session=self.session,
                                   weights=np.array([[[[100.0]]]], dtype=np.float32))

        self.assertEqual(layer._weights.get_shape().as_list(), [3, 3, 2, 16])

    def test_reshape(self):
        convolution_nodes = (4, 4, 8)
        input_p = tf.placeholder("float", (None, 20, 20, 3))
        input_vals = np.random.normal(size=(1, 20, 20, 3)).astype(np.float32)
        layer = ConvolutionalLayer(InputLayer(input_p), convolution_nodes, session=self.session)

        result1 = self.session.run(layer.activation_predict, feed_dict={layer.input_placeholder: input_vals})

        layer.resize(9)
        result2 = self.session.run(layer.activation_predict, feed_dict={layer.input_placeholder: input_vals})

        print(result1)
        print(result2)

        self.assertEquals(result2.shape[3], 9)

    def test_create_extra_weight_dimensions_fail_case(self):
        input_p = tf.placeholder("float", (None, 10, 10, 3))
        layer = ConvolutionalLayer(InputLayer(input_p), (2, 2, 4), session=self.session,
                                   weights=np.random.normal(size=(2, 2, 1, 1)).astype(np.float32))

        self.assertEqual(layer._weights.get_shape().as_list(), [2, 2, 3, 4])

    def test_resize(self):
        convolution_nodes = (4, 4, 8)
        input_p = tf.placeholder("float", (None, 20, 20, 3))
        layer = ConvolutionalLayer(InputLayer(input_p), convolution_nodes, session=self.session)
        layer.resize(9)

        print layer._bias.get_shape()

        self.assertEqual(layer.activation_predict.get_shape().as_list(), [None, 20, 20, 9])
        self.assertEquals(layer.output_nodes, (20, 20, 9))

    def test_get_output_layer_activation(self):
        input_p = tf.placeholder("float", (None, 10, 10, 3))
        layer = ConvolutionalLayer(InputLayer(input_p), (4, 4, 4), session=self.session)
        layer2 = ConvolutionalLayer(layer, (2, 2, 8), session=self.session)
        layer3 = ConvolutionalLayer(layer2, (2, 2, 16), session=self.session)

        self.assertEquals(layer.last_layer.activation_predict, layer3.activation_predict)

    # def test_layer_noisy_input_activation(self):
    #     input_size = 100
    #     noise_std = 1.
    #     input_p = tf.placeholder("float", (None, input_size))
    #     layer = ConvolutionalLayer(InputLayer(input_p), input_size,
    #                                weights=np.diag(np.ones(input_size, dtype=np.float32)),
    #                                bias=np.zeros(input_size, dtype=np.float32),
    #                                session=self.session,
    #                                non_liniarity=tf.identity)
    #
    #     result_noisy = self.session.run(layer.activation_train,
    #                                     feed_dict={
    #                                         input_p: np.ones(input_size, dtype=np.float32).reshape((1, input_size))})
    #
    #     self.assertAlmostEqual(result_noisy.std(), noise_std, delta=noise_std / 5.,
    #                            msg="the result std should be the noise_std")
    #
    #     layer.predict = True
    #
    #     result_clean = self.session.run(layer.activation_predict, feed_dict={
    #         input_p: np.ones(input_size, dtype=np.float32).reshape((1, input_size))})
    #
    #     self.assertAlmostEqual(result_clean.std(), 0., places=7,
    #                            msg="There should be no noise in the activation")
