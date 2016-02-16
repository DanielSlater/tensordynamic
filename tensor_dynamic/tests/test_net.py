from unittest import TestCase
import tensorflow as tf
import numpy as np

from tensor_dynamic.net import Net


class TestNet(TestCase):

    def setUp(self):
        self.session = tf.Session()
        self.session.__enter__()

    def tearDown(self):
        self.session.__exit__(None, None, None)

    def test_hidden_nodes(self):
        input_p = tf.placeholder("float", (None, 3))
        target_p = tf.placeholder("float", (None, 2))
        net = Net(input_p, target_p)
        net.add_outer_layer(5)
        net.add_outer_layer(3)

        self.assertEquals(net.hidden_nodes, [5, 3])