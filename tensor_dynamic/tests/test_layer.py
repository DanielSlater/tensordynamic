from unittest import TestCase
import tensorflow as tf
import numpy as np

from tensor_dynamic.input_layer import InputLayer
from tensor_dynamic.layer import Layer
from tensor_dynamic.tests.base_tf_testcase import BaseTfTestCase


class TestLayer(BaseTfTestCase):
    def setUp(self):
        self.session = tf.Session()
        self.session.__enter__()
        self.session.as_default().__enter__()

    def tearDown(self):
        self.session.__exit__(None, None, None)

    def test_create_layer(self):
        output_nodes = 20
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session)

        self.assertEqual(layer.activation.get_shape().as_list(), [None, output_nodes])

    def test_create_extra_weight_dimensions(self):
        output_nodes = 2
        input_p = tf.placeholder("float", (None, 2))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session,
                                 weights=np.array([[100.0]], dtype=np.float32))

        self.assertEqual(layer._weights.get_shape().as_list(), [2, 2])

    def test_reshape(self):
        output_nodes = 2
        input_p = tf.placeholder("float", (None, 2))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session,
                                 weights=np.array([[100.0]], dtype=np.float32))

        result1 = self.session.run(layer.activation, feed_dict={layer.input_placeholder: [[1., 1.]]})

        layer.resize(3)
        result2 = self.session.run(layer.activation, feed_dict={layer.input_placeholder: [[1., 1.]]})

        print(result1)
        print(result2)

        self.assertEquals(len(result2[0]), 3)

    def test_create_extra_weight_dimensions_fail_case(self):
        output_nodes = 2
        input_p = tf.placeholder("float", (None, 4))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session,
                                 weights=np.array([[10., 10.],
                                                   [10., 10.],
                                                   [10., 10.]], dtype=np.float32))

        self.assertEqual(layer._weights.get_shape().as_list(), [4, 2])

    def test_resize(self):
        output_nodes = 10
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(InputLayer(input_p), output_nodes, session=self.session)
        layer.resize(output_nodes + 1)

        print layer._bias.get_shape()

        self.assertEqual(layer.activation.get_shape().as_list(), [None, output_nodes + 1])
        self.assertEquals(layer.output_nodes, output_nodes + 1)

    def test_get_layers_list(self):
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(InputLayer(input_p), 1, session=self.session)
        layer2 = Layer(layer, 2, session=self.session)
        layer3 = Layer(layer2, 3, session=self.session)

        self.assertEquals(layer.get_layers_list(), [layer, layer2, layer3])

    def test_get_output_layer_activation(self):
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(InputLayer(input_p), 1, session=self.session)
        layer2 = Layer(layer, 2, session=self.session)
        layer3 = Layer(layer2, 3, session=self.session)

        self.assertEquals(layer.output_layer_activation, layer3.activation)
