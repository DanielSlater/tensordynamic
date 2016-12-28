import numpy as np
import tensorflow as tf

from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tests.layers.base_layer_testcase import BaseLayerWrapper


class TestBatchNormLayer(BaseLayerWrapper.BaseLayerTestCase):
    def _create_layer_for_test(self):
        return BatchNormLayer(self._input_layer, self.session)

    def test_normalize(self):
        samples = 20
        input_nodes = 20
        input = InputLayer(input_nodes)
        batchLayer = BatchNormLayer(input, self.session)

        data = np.random.normal(200., 100., size=(samples, input_nodes))

        result = self.session.run(batchLayer.activation_train,
                                  feed_dict={batchLayer.input_placeholder:
                                                 data})

        self.assertAlmostEqual(result.mean(), 0., 3)
        self.assertAlmostEqual(result.var(), 1., 3)

    def test_predict_after_training(self):
        samples = 200
        input_nodes = 2
        input = InputLayer(input_nodes)
        batch_norma_layer = BatchNormLayer(input, self.session)

        # add the updates of batch normalization statistics to train_step
        train = batch_norma_layer.activation_train
        # with tf.control_dependencies([train]):
        #     train = tf.group(batch_norma_layer.assign_op)

        for i in range(200):
            data = np.random.normal(200., 10., size=(samples, input_nodes))
            self.session.run(train,
                             feed_dict={batch_norma_layer.input_placeholder:
                                            data})

        data2 = np.random.normal(200., 10., size=(samples, input_nodes))

        result = self.session.run(batch_norma_layer.activation_predict,
                                  feed_dict={batch_norma_layer.input_placeholder:
                                                 data2})

        self.assertAlmostEqual(result.mean(), 0., delta=10.)
        self.assertAlmostEqual(result.var(), 1., delta=1.)

    def test_resize(self):
        # batch norm layer is resized based only on it's input layer
        input_nodes = 2
        input = InputLayer(input_nodes)
        layer = Layer(input, 2, self.session)
        batchLayer = BatchNormLayer(layer, self.session)

        RESIZE_NODES = 3
        layer.resize(RESIZE_NODES)

        self.assertEqual(batchLayer.output_nodes, RESIZE_NODES)

        self.session.run(batchLayer.activation_predict, feed_dict={batchLayer.input_placeholder: [np.ones(2)]})

    def test_predict_vs_train_bn(self):
        data = self.mnist_data
        bn = BatchNormLayer(InputLayer(784), session=self.session)

        optimizer = bn.activation_train

        # add the updates of batch normalization statistics to train_step
        # with tf.control_dependencies([optimizer]):
        #     optimizer = tf.group(bn.assign_op)

        self.session.run(tf.initialize_all_variables())

        end_epoch = data.train.epochs_completed + 3

        while data.train.epochs_completed <= end_epoch:
            train_x, train_y = data.train.next_batch(100)
            self.session.run(optimizer, feed_dict={bn.input_placeholder: train_x})

        result_predict = self.session.run(bn.activation_predict,
                                          feed_dict={bn.input_placeholder: data.test.images})

        result_train = self.session.run(bn.activation_train,
                                        feed_dict={bn.input_placeholder: data.test.images})

        self.assertAlmostEqual(result_train.mean(), result_predict.mean(), delta=0.2)
        self.assertAlmostEqual(result_train.var(), result_predict.var(), delta=0.2)

    def test_predict_vs_train_similar_activation(self):
        data = self.mnist_data
        bn = BatchNormLayer(InputLayer(784), session=self.session)
        layer = Layer(bn, 5, session=self.session, bactivate=True)

        cost = layer.unsupervised_cost_train()
        optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

        # add the updates of batch normalization statistics to train_step
        # with tf.control_dependencies([optimizer]):
        #     optimizer = tf.group(bn._assign_op)

        self.session.run(tf.initialize_all_variables())

        end_epoch = data.train.epochs_completed + 3

        while data.train.epochs_completed <= end_epoch:
            train_x, train_y = data.train.next_batch(100)
            self.session.run(optimizer, feed_dict={layer.input_placeholder: train_x})

        result_train = self.session.run(layer.unsupervised_cost_train(),
                                        feed_dict={layer.input_placeholder: data.test.images})
        result_predict = self.session.run(layer.unsupervised_cost_predict(),
                                          feed_dict={layer.input_placeholder: data.test.images})

        self.assertAlmostEqual(result_train, result_predict, delta=0.2)
