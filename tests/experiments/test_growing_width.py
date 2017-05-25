import tensorflow as tf
from tensor_dynamic.bayesian_resizing_net import BasicResizableNetWrapper, BayesianResizingNet, EDataType
from tensor_dynamic.utils import get_model_parameters
from tests.base_tf_testcase import get_mnist_data, BaseTfTestCase


def _create_resizing_net(self, dimensions):
    inner_net = BasicResizableNetWrapper(dimensions, self.session)
    outer_net = BayesianResizingNet(inner_net, model_selection_data_type=EDataType.TEST)
    return outer_net


def main():
    data_set = get_mnist_data()

    results = []

    with tf.Session() as session:
        net = BasicResizableNetWrapper((BaseTfTestCase.MNIST_INPUT_NODES, 30, BaseTfTestCase.MNIST_OUTPUT_NODES),
                                       session, regularizer_coeff=1e-4, model_selection_data_type=EDataType.TEST)

        error = net.train_till_convergence(data_set)
        parameters = get_model_parameters(net.get_dimensions())
        results.append((net.get_dimensions()[1], parameters, error))

        while net.get_dimensions()[1] < 1200:
            net.resize_layer(1, int(net.get_dimensions()[1] * 1.1), data_set)
            error = net.train_till_convergence(data_set)
            parameters = get_model_parameters(net.get_dimensions())
            results.append((net.get_dimensions()[1], parameters, error))

    print results


if __name__ == '__main__':
    main()
