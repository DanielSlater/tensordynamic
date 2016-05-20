import numpy as np
import tensorflow as tf

from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.tests.base_layer_testcase import BaseLayerWrapper
from tensor_dynamic.tests.base_tf_testcase import BaseTfTestCase


class TestLayer(BaseLayerWrapper.BaseLayerTestCase):
    def _create_layer_for_test(self):
        return Layer(self._input_layer, self.OUTPUT_NODES, session=self.session)

    def test_create_layer(self):
        output_nodes = 20
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session)

        self.assertEqual(layer.activation_predict.get_shape().as_list(), [None, output_nodes])

    def test_create_extra_weight_dimensions(self):
        output_nodes = 2
        input_p = tf.placeholder("float", (None, 2))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session,
                      weights=np.array([[100.0]], dtype=np.float32))

        self.assertEqual(layer._weights.get_shape().as_list(), [2, 2])

    def test_reshape(self):
        output_nodes = 2
        input_p = tf.placeholder("float", (None, 2))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session,
                      weights=np.array([[100.0]], dtype=np.float32))

        result1 = self.session.run(layer.activation_predict, feed_dict={layer.input_placeholder: [[1., 1.]]})

        layer.resize(3)
        result2 = self.session.run(layer.activation_predict, feed_dict={layer.input_placeholder: [[1., 1.]]})

        print(result1)
        print(result2)

        self.assertEquals(len(result2[0]), 3)

    def test_create_extra_weight_dimensions_fail_case(self):
        output_nodes = 2
        input_p = tf.placeholder("float", (None, 4))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session,
                      weights=np.array([[10., 10.],
                                        [10., 10.],
                                        [10., 10.]], dtype=np.float32))

        self.assertEqual(layer._weights.get_shape().as_list(), [4, 2])

    def test_resize(self):
        output_nodes = 10
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session)
        layer.resize(output_nodes + 1)

        print layer._bias.get_shape()

        self.assertEqual(layer.activation_predict.get_shape().as_list(), [None, output_nodes + 1])
        self.assertEquals(layer.output_nodes, output_nodes + 1)

    def test_get_layers_list(self):
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(InputLayer(input_p), 1, session=self.session)
        layer2 = Layer(layer, 2, session=self.session)
        layer3 = Layer(layer2, 3, session=self.session)

        self.assertEquals(layer.get_layers_list(), [layer, layer2, layer3])

    def test_get_output_layer_activation(self):
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(InputLayer(input_p), 1, session=self.session)
        layer2 = Layer(layer, 2, session=self.session)
        layer3 = Layer(layer2, 3, session=self.session)

        self.assertEquals(layer.last_layer.activation_predict, layer3.activation_predict)

    def test_layer_noisy_input_activation(self):
        input_size = 100
        noise_std = 1.
        input_p = tf.placeholder("float", (None, input_size))
        layer = Layer(InputLayer(input_p), input_size,
                      weights=np.diag(np.ones(input_size, dtype=np.float32)),
                      bias=np.zeros(input_size, dtype=np.float32),
                      session=self.session,
                      non_liniarity=tf.identity,
                      noise_std=noise_std)

        result_noisy = self.session.run(layer.activation_train,
                                        feed_dict={
                                            input_p: np.ones(input_size, dtype=np.float32).reshape((1, input_size))})

        self.assertAlmostEqual(result_noisy.std(), noise_std, delta=noise_std / 5.,
                               msg="the result std should be the noise_std")

        layer.predict = True

        result_clean = self.session.run(layer.activation_predict, feed_dict={
            input_p: np.ones(input_size, dtype=np.float32).reshape((1, input_size))})

        self.assertAlmostEqual(result_clean.std(), 0., places=7,
                               msg="There should be no noise in the activation")

    def test_layer_noisy_input_bactivation(self):
        input_size = 100
        noise_std = 1.
        input_p = tf.placeholder("float", (None, input_size))
        layer = Layer(InputLayer(input_p), input_size,
                      weights=np.diag(np.ones(input_size, dtype=np.float32)),
                      bias=np.zeros(input_size, dtype=np.float32),
                      back_bias=np.zeros(input_size, dtype=np.float32),
                      session=self.session,
                      bactivate=True,
                      non_liniarity=tf.identity,
                      noise_std=noise_std)

        result_noisy = self.session.run(layer.bactivation_train,
                                        feed_dict={
                                            input_p: np.ones(input_size, dtype=np.float32).reshape((1, input_size))})

        self.assertAlmostEqual(result_noisy.std(), noise_std, delta=noise_std / 4.,
                               msg="the result std should be the noise_std")

        result_clean = self.session.run(layer.bactivation_predict, feed_dict={
            input_p: np.ones(input_size, dtype=np.float32).reshape((1, input_size))})

        self.assertAlmostEqual(result_clean.std(), 0., delta=0.1,
                               msg="When running in prediction mode there should be no noise in the bactivation")
