import logging
import os

import numpy as np
from unittest import TestCase

import sys
import tensorflow as tf

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def get_mnist_data(limit_size=None, flatten=True):
    import tensor_dynamic.data.mnist_data as mnist
    import tensor_dynamic.data.data_set as ds
    import os
    return mnist.get_mnist_data_set_collection(os.path.dirname(ds.__file__) + BaseTfTestCase.MNIST_DATA_DIR, one_hot=True,
                                               flatten=flatten,
                                               limit_train_size=limit_size)

class BaseTfTestCase(TestCase):
    MNIST_DATA = None
    MNIST_INPUT_NODES = 784
    MNIST_OUTPUT_NODES = 10
    MNIST_LIMIT_TEST_DATA_SIZE = 1000
    MNIST_DATA_DIR = "/MNIST_data"

    def setUp(self):
        self.session = tf.Session()
        self.session.__enter__()
        self.session.as_default().__enter__()

    def tearDown(self):
        self.session.__exit__(None, None, None)

    @property
    def mnist_data(self):
        if self.MNIST_DATA is None:
            self.MNIST_DATA = get_mnist_data(limit_size=self.MNIST_LIMIT_TEST_DATA_SIZE)
        return self.MNIST_DATA

    def data_sum_of_gaussians(self, num_guassians, data_width, data_count):
        gauss_to_data = np.random.uniform(-1., 1., size=(num_guassians, data_width))
        data = []
        for i in range(data_count):
            guassians = np.random.normal(size=num_guassians)
            data_item = np.matmul(guassians, gauss_to_data)
            data.append(data_item)

        return data
