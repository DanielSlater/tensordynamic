from tensor_dynamic.bayesian_resizing_net import BasicResizableNetWrapper, BayesianResizingNet
from tests.base_tf_testcase import BaseTfTestCase


class TestBayesianResizingNet(BaseTfTestCase):
    def _create_resizing_net(self, dimensions):
        inner_net = BasicResizableNetWrapper(dimensions, self.session)
        outer_net = BayesianResizingNet(inner_net)
        return outer_net

    def test_shrink_from_too_big(self):
        net = self._create_resizing_net((self.MNIST_INPUT_NODES, 1100, self.MNIST_OUTPUT_NODES))
        net.run(self.mnist_data)

        print net._resizable_net.get_dimensions()

        self.assertLess(net._resizable_net.get_dimensions()[1], 1000)

    def test_grow_from_too_small(self):
        # does not always pass
        net = self._create_resizing_net((self.MNIST_INPUT_NODES, 5, self.MNIST_OUTPUT_NODES))
        net.run(self.mnist_data)

        print net._resizable_net.get_dimensions()

        self.assertGreater(net._resizable_net.get_dimensions()[1], 10)