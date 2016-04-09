import tensorflow as tf
import numpy as np
from tensor_dynamic.data_functions import XOR_INPUTS, XOR_TARGETS
from tensor_dynamic.input_layer import InputLayer, NoisyInputLayer
from tensor_dynamic.ladder_layer import LadderLayer
from tensor_dynamic.net import Net
from tensor_dynamic.tests.test_layer import TestLayer


class TestLadderLayer(TestLayer):
    def setUp(self):
        super(TestLadderLayer, self).setUp()
        self.LAYER_CLASS = LadderLayer

    def test_batch_normalize(self):
        inputs = tf.placeholder("float", (None, 2))
        batch_norm_op = LadderLayer.batch_normalization(inputs)

        self.assertTrue(np.array_equal(self.session.run(batch_norm_op, feed_dict={inputs: [[1.0, 1.0]]}), [[0.0, 0.0]]))
        self.assertTrue(np.array_equal(self.session.run(batch_norm_op, feed_dict={inputs: [[1.0, 1.0], [0.0, -1.0]]}),
                                       [[1., 1.], [-1., -1.]]))

    def test_train(self):
        inputs = tf.placeholder("float", (None, 2))
        targets = tf.placeholder("float", (None, 1))
        layer = LadderLayer(inputs, 1, self.session)
        print layer.activation
        inputs_vals = [[0.1, 1.0], [-0.2, 0.6]]

        print("z_pre ", self.session.run(layer.z_pre, feed_dict={inputs: inputs_vals}))
        print("z_bn ", self.session.run(layer.z, feed_dict={inputs: inputs_vals}))
        print("z clean ", self.session.run(layer.activation, feed_dict={inputs: inputs_vals}))
        print("z corr ", self.session.run(layer.activation_noisy, feed_dict={inputs: inputs_vals}))
        result = self.session.run(layer.activation_noisy, feed_dict={inputs: [[0.0, 1.0], [0.0, 0.0]]})
        print("corrupted ", result)
        result = self.session.run(layer.bactivation, feed_dict={inputs: [[0.0, 1.0], [0.0, 0.0]]})
        print("bactivation ", result)

        result = self.session.run(layer.z_est_bn, feed_dict={inputs: [[0.0, 1.0], [0.0, 0.0]]})
        print("z_est_bn ", result)

        result = self.session.run([layer.mean, layer.variance], feed_dict={inputs: [[0.0, 1.0], [0.5, 0.5]]})
        print("mean, var ", result)

        result = self.session.run(layer.get_supervised_cost(targets),
                                  feed_dict={
                                      inputs: [[0.0, 1.0], [0.0, 0.0]],
                                      targets: [[0.0], [1.0]]})
        print("cost ", result)

        cost = layer.get_cost(targets)
        adamOptimizer = tf.train.AdamOptimizer(0.01)
        train_step = adamOptimizer.minimize(cost)

        self.session.run(tf.initialize_all_variables())

        for i in range(100):
            _, a, b = self.session.run([train_step, layer.activation, layer.bactivation], feed_dict={
                inputs: [[0.0, 1.0], [-1.0, 0.5]],
                targets: [[1.0], [-1.0]]
            })

            print a, b

    def test_bactivation(self):
        placeholder = tf.placeholder("float", (None, 4))
        layer = LadderLayer(NoisyInputLayer(placeholder, self.session), 2, 0.1, self.session)
        self.assertEquals([None, 4], layer.bactivation.get_shape().as_list())

    def test_train_xor(self):
        train_x = [[0.0, 1.0, -1.0, 0.0],
                   [1.0, 0.0, -1.0, 1.0],
                   [0.0, 1.0, -1.0, -1.0],
                   [-1.0, 0.5, 1.0, 0.0]]
        train_y = [[-1.0, 0.0],
                   [1.0, 1.0],
                   [0., -1.0],
                   [-1.0, 0.0]]
        net = Net(self.session, 4, 2, layer_class=self.LAYER_CLASS)

        loss_1 = net.train(train_x, train_y, batch_size=2)
        loss_2 = net.train(train_x, train_y, batch_size=2)
        print net.train(train_x, train_y, batch_size=2)
        print net.train(train_x, train_y, batch_size=2)
        print net.train(train_x, train_y, batch_size=2)

        self.assertGreater(loss_1, loss_2, msg="Expected loss to reduce")