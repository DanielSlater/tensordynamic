import tensorflow as tf
from tensor_dynamic.bayesian_resizing_net import create_flat_network
from tests.base_tf_testcase import get_mnist_data


def main(data_set_collection):
    results = []

    with tf.Session() as session:
        net = create_flat_network((data_set_collection.features_shape[0],
                                   20,
                                   data_set_collection.labels_shape[0]), session)

        error = net.train_till_convergence(data_set_collection.train, data_set_collection.test,
                                           learning_rate=0.001)
        parameters = net.get_parameters_all_layers()
        results.append((net.get_resizable_dimensions()[0], parameters, error))

        while net.get_resizable_dimensions()[0] <= 500:
            net.get_all_resizable_layers()[0].resize(net.get_resizable_dimensions()[0] + 10)
            error = net.train_till_convergence(data_set_collection.train, data_set_collection.test,
                                               learning_rate=0.0001)
            parameters = net.get_parameters_all_layers()
            results.append((net.get_resizable_dimensions()[0], parameters, error))

    print results


if __name__ == '__main__':
    data_set_collection = get_mnist_data()
    main(data_set_collection)
