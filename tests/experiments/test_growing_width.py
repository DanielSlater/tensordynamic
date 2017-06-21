import tensorflow as tf
from tensor_dynamic.bayesian_resizing_net import BasicResizableNetWrapper, BayesianResizingNet, EDataType, \
    create_network
from tests.base_tf_testcase import get_mnist_data, BaseTfTestCase


def _create_resizing_net(self, dimensions):
    inner_net = BasicResizableNetWrapper(create_network(dimensions, self.session),
                                         model_selection_data_type=EDataType.TEST)
    outer_net = BayesianResizingNet(inner_net)
    return outer_net


def main():
    data_set = get_mnist_data()

    results = []

    with tf.Session() as session:
        net = BasicResizableNetWrapper(create_network((BaseTfTestCase.MNIST_INPUT_NODES, 30, BaseTfTestCase.MNIST_OUTPUT_NODES),
                                       session, regularizer_coeff=1e-4), model_selection_data_type=EDataType.TEST)

        error = net.train_till_convergence(data_set)
        parameters = net.get_parameters_all_layers()
        results.append((net.get_resizable_dimensions()[0], parameters, error))

        while net.get_resizable_dimensions()[1] < 1200:
            net.resize_layer(1, int(net.get_resizable_dimensions()[0] * 1.1), data_set)
            error = net.train_till_convergence(data_set)
            parameters = net.get_parameters_all_layers()
            results.append((net.get_resizable_dimensions()[0], parameters, error))

    print results


if __name__ == '__main__':
    main()
