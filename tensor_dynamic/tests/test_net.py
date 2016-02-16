from unittest import TestCase
import tensorflow as tf
import numpy as np

from tensor_dynamic.net import Net


class TestNet(TestCase):

    def setUp(self):
        self.session = tf.Session()
        self.session.__enter__()
        self.input_p = tf.placeholder("float", (None, 3))
        self.target_p = tf.placeholder("float", (None, 2))
        self.net = Net(self.input_p, self.target_p)

    def tearDown(self):
        self.session.__exit__(None, None, None)

    def test_hidden_nodes(self):
        self.net.add_outer_layer(5)

        self.assertEquals(self.net.hidden_nodes, [5])

    def test_add_outer_layer(self):
        self.net.add_outer_layer(5)
        self.net.add_outer_layer(3)

        self.assertEquals(self.net.hidden_nodes, [5, 3])

    def test_add_node_to_layer(self):
        self.net.add_outer_layer(3, session=self.session)
        self.net.add_outer_layer(2, session=self.session)

        self.net.generate_methods()

        new_net = self.net.add_node_to_layer(self.session, 0)

        self.assertEquals(new_net.hidden_nodes[0], 4)
