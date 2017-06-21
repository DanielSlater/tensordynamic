from tensor_dynamic.bayesian_resizing_net import BayesianResizingNet, EDataType, \
    create_network
from tests.base_tf_testcase import BaseTfTestCase


class TestBayesianResizingNet(BaseTfTestCase):
    MNIST_LIMIT_TEST_DATA_SIZE = 3000

    def _create_resizing_net(self, dimensions):
        outer_net = BayesianResizingNet(create_network(dimensions, self.session),
                                        model_selection_data_type=EDataType.TRAIN)
        return outer_net

    def test_shrink_from_too_big(self):
        net = self._create_resizing_net((self.MNIST_INPUT_NODES, 2000, self.MNIST_OUTPUT_NODES))
        net.run(self.mnist_data)

        print net._output_layer.get_resizable_dimension_size_all_layers()

        self.assertLess(net._output_layer.get_resizable_dimension_size_all_layers()[0], 2000)

    def test_grow_from_too_small(self):
        # does not always pass
        net = self._create_resizing_net((self.MNIST_INPUT_NODES, 5, self.MNIST_OUTPUT_NODES))
        net.run(self.mnist_data)

        print net._output_layer.get_resizable_dimension_size_all_layers()

        self.assertGreater(net._output_layer.get_resizable_dimension_size_all_layers()[0], 10)

    def test_resizing_net_grow(self):
        dimensions = (self.MNIST_INPUT_NODES, 20, self.MNIST_OUTPUT_NODES)
        inner_net = create_network(dimensions, self.session)
        inner_net.train_till_convergence(self.mnist_data.train)
        next(iter(inner_net.get_all_resizable_layers())).resize(25)
        inner_net.train_till_convergence(self.mnist_data.train)

    def test_resizing_net_shrink(self):
        dimensions = (self.MNIST_INPUT_NODES, 20, self.MNIST_OUTPUT_NODES)
        inner_net = create_network(dimensions, self.session)
        inner_net.train_till_convergence(self.mnist_data.train)
        next(iter(inner_net.get_all_resizable_layers())).resize(15)
        inner_net.train_till_convergence(self.mnist_data.train)

    def test_resizing_net_shrink_twice(self):
        dimensions = (self.MNIST_INPUT_NODES, 20, self.MNIST_OUTPUT_NODES)
        inner_net = create_network(dimensions, self.session)
        inner_net.train_till_convergence(self.mnist_data.train)
        next(iter(inner_net.get_all_resizable_layers())).resize(15)
        inner_net.train_till_convergence(self.mnist_data.train)
        next(iter(inner_net.get_all_resizable_layers())).resize(10)
        inner_net.train_till_convergence(self.mnist_data.train)

    def test_loss_does_not_decrease_when_returning_to_old_size_from_small(self):
        dimensions = (self.MNIST_INPUT_NODES, 20, self.MNIST_OUTPUT_NODES)
        inner_net = create_network(dimensions, self.session)
        start_loss = inner_net.train_till_convergence(self.mnist_data.train)
        print("start_loss: ", start_loss)

        next(iter(inner_net.get_all_resizable_layers())).resize(5)
        small_size_loss = inner_net.train_till_convergence(self.mnist_data.train)
        print("small_size_loss: ", small_size_loss)

        next(iter(inner_net.get_all_resizable_layers())).resize(20)
        return_to_old_size_loss = inner_net.train_till_convergence(self.mnist_data.train)
        print("return_to_old_size_loss: ", return_to_old_size_loss)

        self.assertAlmostEqual(start_loss, return_to_old_size_loss, delta=20)

    def test_loss_does_not_decrease_when_returning_to_old_size_from_big(self):
        dimensions = (self.MNIST_INPUT_NODES, 10, self.MNIST_OUTPUT_NODES)
        inner_net = create_network(dimensions, self.session)
        start_loss = inner_net.train_till_convergence(self.mnist_data.train)
        print("start_loss: ", start_loss)

        next(iter(inner_net.get_all_resizable_layers())).resize(50)
        small_size_loss = inner_net.train_till_convergence(self.mnist_data.train)
        print("small_size_loss: ", small_size_loss)

        next(iter(inner_net.get_all_resizable_layers())).resize(10)
        return_to_old_size_loss = inner_net.train_till_convergence(self.mnist_data.train)
        print("return_to_old_size_loss: ", return_to_old_size_loss)

        self.assertAlmostEqual(start_loss, return_to_old_size_loss, delta=20)