import numpy as np
import tensorflow as tf

from tensor_dynamic.categorical_trainer import CategoricalTrainer
from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tests.base_tf_testcase import BaseTfTestCase


class TestNet(BaseTfTestCase):
    def test_resize(self):
        inputs = tf.placeholder(tf.float32, shape=(None, 784))

        bactivate = True
        net1 = InputLayer(inputs)
        net2 = Layer(net1, 10, self.session, bactivate=bactivate)
        bn1 = BatchNormLayer(net2, self.session)
        output_net = Layer(bn1, 10, self.session, bactivate=False)

        print(self.session.run(output_net.activation_predict, feed_dict={inputs: np.zeros(shape=(1, 784))}))

        net2.resize(net2.output_nodes + 1)

        print(self.session.run(output_net.activation_predict, feed_dict={inputs: np.zeros(shape=(1, 784))}))

    def test_resize_deep(self):
        inputs = tf.placeholder(tf.float32, shape=(None, 784))

        bactivate = True
        net1 = InputLayer(inputs)
        bn1 = BatchNormLayer(net1, self.session)
        net2 = Layer(bn1, 8, self.session, bactivate=bactivate)
        bn2 = BatchNormLayer(net2, self.session)
        net2 = Layer(bn2, 6, self.session, bactivate=bactivate)
        bn3 = BatchNormLayer(net2, self.session)
        net3 = Layer(bn3, 4, self.session, bactivate=bactivate)
        output_net = Layer(net3, 2, self.session, bactivate=False)

        print(self.session.run(output_net.activation_predict, feed_dict={inputs: np.zeros(shape=(1, 784))}))

        net2.resize(net2.output_nodes + 1)

        print(self.session.run(output_net.activation_predict, feed_dict={inputs: np.zeros(shape=(1, 784))}))

    def test_layers_with_noise(self):
        inputs = tf.placeholder(tf.float32, shape=(None, 784))

        input_layer = InputLayer(inputs)
        bn1 = BatchNormLayer(input_layer, self.session)
        net1 = Layer(bn1, 70, bactivate=True, noise_std=1.)
        output_net = Layer(net1, 10, bactivate=False, non_liniarity=tf.identity)

        print(self.session.run(output_net.activation_train, feed_dict={inputs: np.zeros(shape=(1, 784))}))

    def test_clone(self):
        inputs = tf.placeholder(tf.float32, shape=(None, 784))

        net1 = InputLayer(inputs)
        bn1 = BatchNormLayer(net1, self.session)
        net2 = Layer(bn1, 8, self.session)
        bn2 = BatchNormLayer(net2, self.session)
        net2 = Layer(bn2, 6, self.session)
        bn3 = BatchNormLayer(net2, self.session)
        net3 = Layer(bn3, 4, self.session)
        output_net = Layer(net3, 2, self.session)

        cloned_net = output_net.clone(self.session)

        self.assertNotEquals(cloned_net, output_net)
        self.assertNotEquals(cloned_net.input_layer, output_net.input_layer)
        self.assertEqual(len(list(cloned_net.all_layers)), len(list(output_net.all_layers)))

    def test_accuracy_bug(self):
        import tensor_dynamic.data.mnist_data as mnist
        data = mnist.read_data_sets("../data/MNIST_data", one_hot=True)

        inputs = tf.placeholder(tf.float32, shape=(None, 784))
        input_layer = InputLayer(inputs)
        outputs = Layer(input_layer, 10, self.session, non_liniarity=tf.sigmoid)

        trainer = CategoricalTrainer(outputs, 0.1)

        trainer.train(data.validation.features, data.validation.labels)

        # this was throwing an exception
        accuracy = trainer.accuracy(data.validation.features, data.validation.labels)
        self.assertLessEqual(accuracy, 100.)
        self.assertGreaterEqual(accuracy, 0.)
