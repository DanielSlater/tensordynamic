import unittest

import numpy as np
import tensorflow as tf
from tensor_dynamic.tests.base_layer_testcase import BaseLayerWrapper

from tensor_dynamic.layers.input_layer import InputLayer, SemiSupervisedInputLayer
from tensor_dynamic.layers.ladder_layer import LadderLayer, LadderGammaLayer
from tensor_dynamic.layers.ladder_output_layer import LadderOutputLayer


class TestLadderLayer(BaseLayerWrapper.BaseLayerTestCase):

    def _create_layer_for_test(self):
        return LadderLayer(SemiSupervisedInputLayer(self.INPUT_NODES), self.OUTPUT_NODES, session=self.session)

    def test_batch_normalize(self):
        inputs = tf.placeholder("float", (None, 2))
        batch_norm_op = LadderLayer.batch_normalization(inputs)

        self.assertTrue(np.array_equal(self.session.run(batch_norm_op, feed_dict={inputs: [[1.0, 1.0]]}), [[0.0, 0.0]]))
        self.assertTrue(np.array_equal(self.session.run(batch_norm_op, feed_dict={inputs: [[1.0, 1.0], [0.0, -1.0]]}),
                                       [[1., 1.], [-1., -1.]]))

    def test_bactivation(self):
        placeholder = tf.placeholder("float", (None, 4))
        input = InputLayer(placeholder, self.session)
        layer = LadderLayer(input, 2, 0.1, self.session)
        LadderOutputLayer(layer, 0.1, self.session)
        self.assertEquals([None, 4], layer.bactivation_predict.get_shape().as_list())
        self.assertEquals([None, 4], layer.bactivation_train.get_shape().as_list())

    @unittest.skip('Need to fix batch sizing for ladder networks')
    def test_train_xor(self):
        train_x = [[0.0, 1.0, -1.0, 0.0],
                   [1.0, 0.0, -1.0, 1.0],
                   [0.0, 1.0, -1.0, -1.0],
                   [-1.0, 0.5, 1.0, 0.0]]
        train_y = [[-1.0, 0.0],
                   [1.0, 1.0],
                   [0., -1.0],
                   [-1.0, 0.0]]
        targets = tf.placeholder('float', (None, 2))

        ladder = InputLayer(len(train_x[0]), self.session)
        ladder = LadderLayer(ladder, 6, 1000., self.session)
        ladder = LadderLayer(ladder, 6, 10., self.session)
        ladder = LadderGammaLayer(ladder, 2, 0.1, self.session)
        ladder = LadderOutputLayer(ladder, 0.1, self.session)

        cost = ladder.cost_all_layers_train(targets)
        train = tf.train.AdamOptimizer(0.1).minimize(cost)

        self.session.run(tf.initialize_all_variables())
        _, cost1 = self.session.run([train, cost], feed_dict={ladder.input_placeholder:train_x, targets:train_y})

        print self.session.run([train, cost], feed_dict={ladder.input_placeholder:train_x, targets:train_y})
        print self.session.run([train, cost], feed_dict={ladder.input_placeholder:train_x, targets:train_y})
        print self.session.run([train, cost], feed_dict={ladder.input_placeholder:train_x, targets:train_y})

        _, cost2 = self.session.run([train, cost], feed_dict={ladder.input_placeholder:train_x, targets:train_y})

        self.assertGreater(cost1, cost2, msg="Expected loss to reduce")

    def test_mnist(self):
        import tensor_dynamic.data.input_data as mnist

        num_labeled = 100
        data = mnist.read_data_sets("../data/MNIST_data", n_labeled=num_labeled, one_hot=True)

        batch_size = 100
        num_epochs = 1
        num_examples = 60000
        num_iter = (num_examples/batch_size) * num_epochs
        starter_learning_rate = 0.02
        inputs = tf.placeholder(tf.float32, shape=(None, 784))
        targets = tf.placeholder(tf.float32)

        with tf.Session() as s:
            s.as_default()
            i = InputLayer(inputs)
            l1 = LadderLayer(i, 500, 1000.0, s)
            l2 = LadderGammaLayer(l1, 10, 10.0, s)
            ladder = LadderOutputLayer(l2, 0.1, s)

            loss = ladder.cost_all_layers_train(targets)
            learning_rate = tf.Variable(starter_learning_rate, trainable=False)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            bn_updates = tf.group(*(l1.bn_assigns + l2.bn_assigns))
            with tf.control_dependencies([train_step]):
                train_step = tf.group(bn_updates)
            pred_cost = -tf.reduce_mean(tf.reduce_sum(targets * tf.log(tf.clip_by_value(ladder.activation_predict, 1e-10, 1.0)), 1))  # cost used for prediction

            correct_prediction = tf.equal(tf.argmax(ladder.activation_predict, 1), tf.argmax(targets, 1))  # no of correct predictions
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

            s.run(tf.initialize_all_variables())

            #print "init accuracy", s.run([accuracy], feed_dict={inputs: data.test.images, targets: data.test.labels})

            min_loss = 100000.

            writer = tf.train.SummaryWriter("/tmp/td", s.graph_def)
            writer.add_graph(s.graph_def)

            for i in range(num_iter):
                images, labels = data.train.next_batch(batch_size)
                _, loss_val = s.run([train_step, loss], feed_dict={inputs: images, targets: labels})

                if loss_val < min_loss:
                    min_loss = loss_val
                print(i, loss_val)

                # print "acc", s.run([accuracy], feed_dict={inputs: data.test.images, targets: data.test.labels})

            #acc = s.run(accuracy, feed_dict={inputs: data.test.images, targets: data.test.labels})
            print "min loss", min_loss
            #print "final accuracy ", acc
            self.assertLess(min_loss, 20.0)
            #self.assertGreater(acc, 70.0)
