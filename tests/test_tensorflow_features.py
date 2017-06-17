import unittest

import numpy as np
import tensorflow as tf

from tensor_dynamic.utils import tf_resize_cascading, tf_resize
from tests.base_tf_testcase import BaseTfTestCase


class TestTensorflowFeatures(BaseTfTestCase):
    def test_extend_dim(self):
        var = tf.Variable(tf.zeros((1,)))
        activation = tf.square(var)

        new_value = tf.zeros((2,))
        change_shape_op = tf.assign(var, new_value, validate_shape=False)

        self.session.run(change_shape_op)  # Changes the shape of `var` to new_value's shape.

        new_var = self.session.run(var)
        new_activation_var = self.session.run(activation)
        self.assertSequenceEqual(new_var.shape, (2,))
        self.assertSequenceEqual(new_activation_var.shape, (2,))

    def test_tf_resize_new_values(self):
        var = tf.Variable(range(20))
        self.session.run(tf.initialize_variables([var]))

        tf_resize(self.session, var, new_values=np.array(range(10)))

        self.assertEqual(len(self.session.run(var)), 10)

    @unittest.skip('functionality not implemented yet')
    def test_cascading_resize(self):
        a = tf.Variable(tf.zeros((1, 2)), name="a")
        b = tf.sigmoid(a, name="b")
        matrix = tf.Variable(tf.zeros((2, 4)), name="matrix")
        y = tf.matmul(b, matrix)

        tf_resize_cascading(self.session, a, np.zeros((1, 3)))
        self.assertSequenceEqual(a.get_shape().as_list(), (3,))
        self.assertSequenceEqual(b.get_shape().as_list(), (3,))
        self.assertSequenceEqual(matrix.get_shape().as_list(), (3, 4))
        self.assertSequenceEqual(y.get_shape().as_list(), (4,))

    def test_resize_convolution_convolution_dimension(self):
        input_var = tf.Variable(np.random.normal(0., 1., (1, 4, 4, 1)).astype(np.float32))
        weights = tf.Variable(np.ones((2, 2, 1, 32), dtype=np.float32))
        output = tf.nn.relu(
            tf.nn.conv2d(input_var, weights, strides=[1, 1, 1, 1],
                         padding="SAME"))

        self.session.run(tf.initialize_variables([input_var, weights]))

        result1 = self.session.run(output)

        tf_resize(self.session, weights, new_values=np.ones([2, 2, 1, 16]))

        result2 = self.session.run(output)
        assert result2.shape == (1, 4, 4, 16)

    def test_resize_convolution_intput_layer(self):
        # resize input layer
        input_var = tf.Variable(np.random.normal(0., 1., (1, 4, 4, 1)).astype(np.float32))
        weights = tf.Variable(np.ones((2, 2, 1, 32), dtype=np.float32))
        output = tf.nn.relu(
            tf.nn.conv2d(input_var, weights, strides=[1, 1, 1, 1],
                         padding="SAME"))

        self.session.run(tf.initialize_variables([input_var, weights]))
        result1 = self.session.run(output)

        tf_resize(self.session, input_var, new_values=np.ones([1, 5, 5, 1]))
        result3 = self.session.run(output)
        print(result3)
        assert result3.shape == (1, 5, 5, 32)
