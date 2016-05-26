from unittest import TestCase
import tensorflow as tf


class BaseTfTestCase(TestCase):
    MNIST_DATA = None

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
            self.MNIST_DATA = mnist.read_data_sets("../data/MNIST_data", one_hot=True)
        return self.MNIST_DATA