import numpy as np
import tensorflow as tf

from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
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

    def test_more_nodes_improves_reconstruction_loss(self):
        recon_1 = self.reconstruction_loss_for(1)
        recon_2 = self.reconstruction_loss_for(2)
        self.assertLess(recon_2, recon_1)
        recon_5 = self.reconstruction_loss_for(5)
        self.assertLess(recon_5, recon_2)
        recon_20 = self.reconstruction_loss_for(20)
        self.assertLess(recon_20, recon_5)

    def reconstruction_loss_for(self, output_nodes):
        data = self.mnist_data
        bw_layer1 = Layer(InputLayer(784), output_nodes, session=self.session, noise_std=1.0, bactivate=True)

        cost = bw_layer1.unsupervised_cost_train()
        optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

        self.session.run(tf.initialize_all_variables())

        end_epoch = data.train.epochs_completed + 3

        while data.train.epochs_completed <= end_epoch:
            train_x, train_y = data.train.next_batch(100)
            self.session.run(optimizer, feed_dict={bw_layer1.input_placeholder: train_x})

        result = self.session.run(bw_layer1.unsupervised_cost_predict(),
                                  feed_dict={bw_layer1.input_placeholder: data.test.images})
        print("denoising with %s hidden layer had cost %s" % (output_nodes, result))
        return result

    def test_reconstruction_of_single_input(self):
        input_layer = InputLayer(1)
        layer = Layer(input_layer, 1, bactivate=True, session=self.session, noise_std=0.3)

        cost = layer.unsupervised_cost_train()
        optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

        self.session.run(tf.initialize_all_variables())

        data = np.random.normal(0.5, 0.5, size=[200, 1])

        for x in range(100):
            self.session.run([optimizer], feed_dict={input_layer.input_placeholder: data})

        result = self.session.run([cost], feed_dict={input_layer.input_placeholder: data})
        print result