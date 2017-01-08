from tensor_dynamic.bayesian_resizing_net import BasicResizableNetWrapper, BayesianResizingNet
from tests.base_tf_testcase import BaseTfTestCase


class TestBayesianResizingNet(BaseTfTestCase):
    def _create_resizing_net(self, dimensions):
        inner_net = BasicResizableNetWrapper(dimensions, self.session)
        outer_net = BayesianResizingNet(inner_net)
        return outer_net

    def test_shrink_from_too_big(self):
        net = self._create_resizing_net((self.MNIST_INPUT_NODES, 2000, self.MNIST_OUTPUT_NODES))
        net.run(self.mnist_data)

        print net._resizable_net.get_dimensions()

        self.assertLess(net._resizable_net.get_dimensions()[1], 2000)

    def test_grow_from_too_small(self):
        # does not always pass
        net = self._create_resizing_net((self.MNIST_INPUT_NODES, 5, self.MNIST_OUTPUT_NODES))
        net.run(self.mnist_data)

        print net._resizable_net.get_dimensions()

        self.assertGreater(net._resizable_net.get_dimensions()[1], 10)

    def test_resizing_net_grow(self):
        dimensions = (self.MNIST_INPUT_NODES, 20, self.MNIST_OUTPUT_NODES)
        inner_net = BasicResizableNetWrapper(dimensions, self.session)
        inner_net.train_till_convergence(self.mnist_data)
        inner_net.resize_layer(1, 25, self.mnist_data)
        inner_net.train_till_convergence(self.mnist_data)

    def test_resizing_net_shrink(self):
        dimensions = (self.MNIST_INPUT_NODES, 20, self.MNIST_OUTPUT_NODES)
        inner_net = BasicResizableNetWrapper(dimensions, self.session)
        inner_net.train_till_convergence(self.mnist_data)
        inner_net.resize_layer(1, 15, self.mnist_data)
        inner_net.train_till_convergence(self.mnist_data)

    def test_resizing_net_shrink_twice(self):
        dimensions = (self.MNIST_INPUT_NODES, 20, self.MNIST_OUTPUT_NODES)
        inner_net = BasicResizableNetWrapper(dimensions, self.session)
        inner_net.train_till_convergence(self.mnist_data)
        inner_net.resize_layer(1, 15, self.mnist_data)
        inner_net.train_till_convergence(self.mnist_data)
        inner_net.resize_layer(1, 10, self.mnist_data)
        inner_net.train_till_convergence(self.mnist_data)