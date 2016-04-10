import tensorflow as tf
import numpy as np

from tensor_dynamic.data_functions import XOR_INPUTS, XOR_TARGETS
from tensor_dynamic.input_layer import InputLayer, NoisyInputLayer
from tensor_dynamic.ladder_layer import LadderLayer, LadderGammaLayer
from tensor_dynamic.ladder_output_layer import LadderOutputLayer
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
        input_layer = NoisyInputLayer(inputs, self.session)
        layer = LadderLayer(input_layer, 1, session=self.session)
        print layer.activation
        inputs_vals = [[0.1, 1.0], [-0.2, 0.6]]

        print("z_pre ", self.session.run(layer.z_pre, feed_dict={inputs: inputs_vals}))
        print("z_bn ", self.session.run(layer.z_corrupted, feed_dict={inputs: inputs_vals}))
        print("z clean ", self.session.run(layer.activation, feed_dict={inputs: inputs_vals}))
        print("z corr ", self.session.run(layer.activation_corrupted, feed_dict={inputs: inputs_vals}))
        result = self.session.run(layer.activation_corrupted, feed_dict={inputs: [[0.0, 1.0], [0.0, 0.0]]})
        print("corrupted ", result)
        result = self.session.run(layer.bactivation, feed_dict={inputs: [[0.0, 1.0], [0.0, 0.0]]})
        print("bactivation ", result)

        result = self.session.run(layer.z_est_bn, feed_dict={inputs: [[0.0, 1.0], [0.0, 0.0]]})
        print("z_est_bn ", result)

        result = self.session.run([layer.mean_corrupted, layer.variance_corrupted], feed_dict={inputs: [[0.0, 1.0], [0.5, 0.5]]})
        print("mean, var ", result)

        result = self.session.run(layer.unsupervised_cost(),
                                  feed_dict={
                                      inputs: [[0.0, 1.0], [0.0, 0.0]]})
        print("cost ", result)

        cost = layer.cost(targets)
        adamOptimizer = tf.train.AdamOptimizer(0.1)
        train_step = adamOptimizer.minimize(cost)

        self.session.run(tf.initialize_all_variables())

        for i in range(100):
            _, cost_val, a, b = self.session.run([train_step, cost, layer.activation, layer.bactivation], feed_dict={
                inputs: [[0.0, 1.0], [-1.0, 0.5]],
                targets: [[1.0], [-1.0]]
            })

            print cost_val

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
        targets = tf.placeholder('float', (None, 2))

        ladder = NoisyInputLayer(len(train_x[0]), self.session)
        ladder = LadderLayer(ladder, 6, 1000.)
        ladder = LadderLayer(ladder, 6, 10.)
        ladder = LadderGammaLayer(ladder, 2, 0.1)
        ladder = LadderOutputLayer(ladder, 0.1)

        cost = ladder.cost(targets)
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
        learning_rate = 0.1
        inputs = tf.placeholder(tf.float32, shape=(None, 784))
        targets = tf.placeholder(tf.float32)

        with tf.Session() as s:
            s.as_default()
            i = InputLayer(inputs)
            l1 = LadderLayer(i, 500, 1000.0, s)
            l2 = LadderGammaLayer(l1, 10, 10.0, s)
            ladder = LadderOutputLayer(l2, 0.1, s)

            loss = ladder.cost_all_layers(targets)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            pred_cost = -tf.reduce_mean(tf.reduce_sum(targets * tf.log(ladder.activation), 1))  # cost used for prediction

            correct_prediction = tf.equal(tf.argmax(ladder.activation, 1), tf.argmax(targets, 1))  # no of correct predictions
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

            s.run(tf.initialize_all_variables())

            ladder.set_all_deterministic(True)

            print "init accuracy", s.run([accuracy], feed_dict={inputs: data.test.images, targets: data.test.labels})

            min_loss = 100000.

            for i in range(num_iter):
                images, labels = data.train.next_batch(batch_size)
                _, loss_val = s.run([train_step, loss], feed_dict={inputs: images, targets: labels})

                if loss_val < min_loss:
                    min_loss = loss_val
                print(i, loss_val)
                # print "acc", s.run([accuracy], feed_dict={inputs: data.test.images, targets: data.test.labels})

            acc = s.run(accuracy, feed_dict={inputs: data.test.images, targets: data.test.labels})
            print "min loss", min_loss
            print "final accuracy ", acc
            self.assertLess(min_loss, 20.0)
            self.assertGreater(acc, 70.0)
