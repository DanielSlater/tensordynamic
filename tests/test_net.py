import pickle
import unittest

import numpy as np
import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
from tensor_dynamic.layers.categorical_output_layer import CategoricalOutputLayer
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tests.base_tf_testcase import BaseTfTestCase


class TestNet(BaseTfTestCase):
    @unittest.skip('Failing because BatchNormLayer is not resizing')
    def test_resize_shallow(self):
        bactivate = True
        net1 = InputLayer(784)
        net2 = HiddenLayer(net1, 10, self.session, bactivate=bactivate)
        bn1 = BatchNormLayer(net2, self.session)
        output_net = HiddenLayer(bn1, 10, self.session, bactivate=False)

        print(self.session.run(output_net.activation_predict, feed_dict={net1.input_placeholder: np.zeros(shape=(1, 784))}))

        net2.resize(net2.output_nodes + 1)

        print(self.session.run(output_net.activation_predict, feed_dict={net1.input_placeholder: np.zeros(shape=(1, 784))}))

    @unittest.skip('Failing because BatchNormLayer is not resizing')
    def test_resize_deep(self):
        bactivate = True
        net1 = InputLayer(784)
        bn1 = BatchNormLayer(net1, self.session)
        net2 = HiddenLayer(bn1, 8, self.session, bactivate=bactivate)
        bn2 = BatchNormLayer(net2, self.session)
        net2 = HiddenLayer(bn2, 6, self.session, bactivate=bactivate)
        bn3 = BatchNormLayer(net2, self.session)
        net3 = HiddenLayer(bn3, 4, self.session, bactivate=bactivate)
        output_net = HiddenLayer(net3, 2, self.session, bactivate=False)

        print(self.session.run(output_net.activation_predict, feed_dict={net1.input_placeholder: np.zeros(shape=(1, 784))}))

        net2.resize(net2.output_nodes + 1)

        print(self.session.run(output_net.activation_predict, feed_dict={net1.input_placeholder: np.zeros(shape=(1, 784))}))

    def test_layers_with_noise(self):
        input_layer = InputLayer(784)
        bn1 = BatchNormLayer(input_layer, self.session)
        net1 = HiddenLayer(bn1, 70, bactivate=True, layer_noise_std=1.)
        output_net = HiddenLayer(net1, 10, bactivate=False, non_liniarity=tf.identity)

        print(self.session.run(output_net.activation_train, feed_dict={
            input_layer.input_placeholder: np.zeros(shape=(1, 784))}))

    def test_clone(self):
        net1 = InputLayer(784)
        bn1 = BatchNormLayer(net1, self.session)
        net2 = HiddenLayer(bn1, 8, self.session)
        bn2 = BatchNormLayer(net2, self.session)
        net2 = HiddenLayer(bn2, 6, self.session)
        bn3 = BatchNormLayer(net2, self.session)
        net3 = HiddenLayer(bn3, 4, self.session)
        output_net = HiddenLayer(net3, 2, self.session)

        cloned_net = output_net.clone(self.session)

        self.assertNotEquals(cloned_net, output_net)
        self.assertNotEquals(cloned_net.input_layer, output_net.input_layer)
        self.assertEqual(len(list(cloned_net.all_layers)), len(list(output_net.all_layers)))

    def test_accuracy_bug(self):
        import tensor_dynamic.data.mnist_data as mnist
        import tensor_dynamic.data.data_set as ds
        import os

        data = mnist.get_mnist_data_set_collection(os.path.dirname(ds.__file__) + "/MNIST_data", one_hot=True)

        input_layer = InputLayer(data.features_shape)
        outputs = CategoricalOutputLayer(input_layer, data.labels_shape, self.session)

        outputs.train_till_convergence(data.test,
                                       learning_rate=0.2, continue_epochs=1)

        # this was throwing an exception
        accuracy = outputs.accuracy(data.test)
        self.assertLessEqual(accuracy, 100.)
        self.assertGreaterEqual(accuracy, 0.)

    def test_save_load_network(self):
        net1 = InputLayer(784)
        net2 = HiddenLayer(net1, 20, self.session)
        output_net = CategoricalOutputLayer(net2, 10, self.session)

        data = output_net.get_network_pickle()

        new_net = BaseLayer.load_network_from_pickle(data, self.session)

        print new_net

    def test_save_load_network_to_disk(self):
        net1 = InputLayer(784)
        net2 = HiddenLayer(net1, 20, self.session)
        output_net = CategoricalOutputLayer(net2, 10, self.session)

        data = output_net.get_network_pickle()

        with open("temp", "w") as f:
            f.write(data)

        new_data = pickle.load(open("temp", "r"))

        new_net = BaseLayer.load_network_from_state(new_data, self.session)

        print new_net