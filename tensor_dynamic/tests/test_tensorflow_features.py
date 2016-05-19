import unittest
from unittest import TestCase
import tensorflow as tf
import numpy as np

from tensor_dynamic.tests.base_tf_testcase import BaseTfTestCase
from tensor_dynamic.utils import tf_resize_cascading


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
