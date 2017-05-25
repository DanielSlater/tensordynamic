import unittest

import numpy as np

import tensorflow as tf

from tensor_dynamic.utils import train_till_convergence, create_hessian_op, tf_resize, \
    get_tf_optimizer_variables
from tests.base_tf_testcase import BaseTfTestCase


class TestUtils(BaseTfTestCase):
    def test_train_till_convergence(self):
        FINAL_ERROR = 3
        errors = [5, 4, 3, 2, 2, 1, 2, 2, FINAL_ERROR]
        errors_iter = iter(errors)

        final_error = train_till_convergence(lambda: next(errors_iter), continue_epochs=3)

        self.assertEqual(final_error, FINAL_ERROR)

    @unittest.skip('functionality not implemented yet')
    def test_compute_hessian(self):
        # this currently fails because I can't get the method to work, tensorflow does not support gradients after
        # doing a reshape/slice op
        n_input = 3
        n_hidden = 2
        n_output = 1
        x_input = tf.placeholder(tf.float32, shape=[None, n_input])
        y_target = tf.placeholder(tf.float32, shape=[None, n_output])

        hidden_weights = tf.Variable(initial_value=tf.truncated_normal([n_input, n_hidden]))
        hidden_biases = tf.Variable(tf.truncated_normal([n_hidden]))
        hidden = tf.sigmoid(tf.matmul(x_input, hidden_weights) + hidden_biases)

        output_weights = tf.Variable(initial_value=tf.truncated_normal([n_hidden, n_output]))
        output_biases = tf.Variable(tf.truncated_normal([n_output]))
        output = tf.nn.softmax(tf.matmul(hidden, output_weights) + output_biases)
        # Define cross entropy loss
        loss = -tf.reduce_sum(y_target * tf.log(output))

        self.session.run(tf.initialize_variables([hidden_weights, hidden_biases, output_weights, output_biases]))
        hessian_op = create_hessian_op(loss, [hidden_weights, hidden_biases, output_weights, output_biases],
                                       self.session)
        result = self.session.run(hessian_op, feed_dict={x_input: np.random.normal(size=(1, n_input)),
                                                         y_target: np.random.normal(size=(1, n_output))})

        print(result)

    @unittest.skip('functionality not implemented yet')
    def test_compute_hessian_1_variable(self):
        # this currently fails because I can't get the method to work, tensorflow does not support gradients after
        # doing a reshape/slice op
        n_input = 2
        n_output = 2

        x_input = tf.placeholder(tf.float32, shape=[None, n_input])
        y_target = tf.placeholder(tf.float32, shape=[None, n_output])

        weights = tf.Variable(initial_value=[[-2., -1.], [1., 2.]])

        output = tf.nn.softmax(tf.matmul(x_input, weights))
        # Define cross entropy loss
        loss = -tf.reduce_sum(y_target * tf.log(output))

        self.session.run(tf.initialize_variables([weights]))
        hessian_op = create_hessian_op(loss, [weights], self.session)
        result = self.session.run(hessian_op, feed_dict={x_input: np.ones((1, n_input)),
                                                         y_target: np.ones((1, n_output))})

        # TODO D.S find simple hessian example + numbers

        print(result)

    def test_gradient_through_reshape(self):
        input = tf.Variable(initial_value=tf.zeros([2, 2]))
        after_reshape = tf.reshape(input, [-1])
        target = tf.square(1. - tf.reduce_sum(after_reshape))

        train_op = tf.train.GradientDescentOptimizer(0.5).minimize(target)

        self.session.run(tf.initialize_variables([input]))

        print self.session.run(input)

        self.session.run(train_op)

        print self.session.run(input)

        print self.session.run(tf.gradients(target, input))

    def test_tf_resize_shrink(self):
        zeros = tf.zeros((6,))
        var = tf.Variable(initial_value=zeros)
        self.session.run(tf.initialize_variables([var]))

        tf_resize(self.session, var, new_dims=(4,))

        self.assertEqual(self.session.run(var).shape, (4,))

    def test_tf_resize_shrink_twice(self):
        zeros = tf.zeros((6,))
        var = tf.Variable(initial_value=zeros)
        self.session.run(tf.initialize_variables([var]))

        tf_resize(self.session, var, new_dims=(4,))

        tf.train.GradientDescentOptimizer(0.1).minimize(var)

        tf_resize(self.session, var, new_dims=(2,))

        self.assertEqual(self.session.run(var).shape, (2,))

        # this was causing an exception
        tf.train.GradientDescentOptimizer(0.1).minimize(var)

    def test_tf_resize_grow(self):
        zeros = tf.zeros((3,))
        var = tf.Variable(initial_value=zeros)
        self.session.run(tf.initialize_variables([var]))

        tf_resize(self.session, var, new_dims=(6,))

        self.assertEqual(self.session.run(var).shape, (6,))

    def test_tf_resize(self):
        zeros = tf.zeros((4,))
        var = tf.Variable(initial_value=zeros)
        loss = tf.square(1 - tf.reduce_sum(var))
        self.session.run(tf.initialize_variables([var]))

        optimizer_1 = tf.train.RMSPropOptimizer(0.01)
        train_1 = optimizer_1.minimize(loss)
        self.session.run(tf.initialize_variables(list(get_tf_optimizer_variables(optimizer_1))))

        self.session.run(train_1)

        tf_resize(self.session, var, new_dims=(6,))

        optimizer_2 = tf.train.RMSPropOptimizer(0.01)
        train_2 = optimizer_2.minimize(loss)
        self.session.run(tf.initialize_variables(list(get_tf_optimizer_variables(optimizer_2))))

        self.session.run(train_2)
