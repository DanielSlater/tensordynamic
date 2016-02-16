from unittest import TestCase
import tensorflow as tf
import numpy as np
from tensor_dynamic.layer import Layer


class TestLayer(TestCase):

    def setUp(self):
        self.session = tf.Session()
        self.session.__enter__()

    def tearDown(self):
        self.session.__exit__(None, None, None)

    def test_create_layer(self):
        output_nodes = 20
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(input_p, output_nodes, session=self.session)

        self.assertEqual(layer.activation.get_shape().as_list(), [None, output_nodes])

    def test_create_extra_weight_dimensions(self):
        output_nodes = 2
        input_p = tf.placeholder("float", (None, 2))
        layer = Layer(input_p, output_nodes, session=self.session,
                      weights=np.array([[100.0]], dtype=np.float32))

        self.assertEqual(layer._weights.get_shape().as_list(), [2, 2])

    def test_add_output(self):
        output_nodes = 10
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(input_p, output_nodes, session=self.session)
        new_layer = layer.add_output(self.session)

        self.assertEqual(new_layer.activation.get_shape().as_list(), [None, output_nodes+1])

    def test_get_layers_list(self):
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(input_p, 1, session=self.session)
        layer2 = Layer(layer, 2, session=self.session)
        layer3 = Layer(layer2, 3, session=self.session)

        self.assertEquals(layer.get_layers_list(), [layer, layer2, layer3])

    def test_get_output_layer_activation(self):
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(input_p, 1, session=self.session)
        layer2 = Layer(layer, 2, session=self.session)
        layer3 = Layer(layer2, 3, session=self.session)

        self.assertEquals(layer.get_output_layer_activation(), layer3.activation)