from unittest import TestCase
import tensorflow as tf
import numpy as np


class TestTensorflowFeatures(TestCase):
    def test_extend_dim(self):
        with tf.Session() as session:
            var = tf.Variable(tf.zeros((1,)))

            new_value = tf.zeros((2,))
            change_shape_op = tf.assign(var, new_value, validate_shape=False)

            session.run(change_shape_op)  # Changes the shape of `var` to new_value's shape.

            print session.run(var)