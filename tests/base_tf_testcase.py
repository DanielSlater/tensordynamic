import numpy as np
from unittest import TestCase
import tensorflow as tf


class BaseTfTestCase(TestCase):
    MNIST_DATA = None
    MNIST_INPUT_NODES = 784
    MNIST_OUTPUT_NODES = 10

    def setUp(self):
        self.session = tf.Session()
        self.session.__enter__()
        self.session.as_default().__enter__()

    def tearDown(self):
        self.session.__exit__(None, None, None)

    @property
    def mnist_data(self):
        if self.MNIST_DATA is None:
            import tensor_dynamic.data.input_data as mnist
            self.MNIST_DATA = mnist.read_data_sets("../../tensor_dynamic/data/MNIST_data/", one_hot=True)
        return self.MNIST_DATA

    def data_sum_of_gaussians(self, num_gaussians, data_width, data_count):
        gauss_to_data = np.random.uniform(-1., 1., size=(num_gaussians, data_width))
        data = []
        for i in range(data_count):
            gaussians = np.random.normal(size=num_gaussians)
            data_item = np.matmul(gaussians, gauss_to_data)
            data.append(data_item)

        return data
