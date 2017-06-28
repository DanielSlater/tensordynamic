from tensor_dynamic.data.mnist_data import get_mnist_data_set_collection
import tensorflow as tf

# set up network
from tensor_dynamic.layers.categorical_output_layer import CategoricalOutputLayer
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.layers.input_layer import InputLayer

data_set_collection = get_mnist_data_set_collection()

START_SIZE = 30
END_SIZE = 31

with tf.Session() as session:
    non_liniarity = tf.nn.relu

    regularizer_coeff = 0.001
    last_layer = InputLayer(data_set_collection.features_shape,
                            # drop_out_prob=.5,
                            # layer_noise_std=1.
                            )

    for _ in range(1):
        last_layer = HiddenLayer(last_layer, START_SIZE, session, non_liniarity=non_liniarity,
                                 batch_normalize_input=True)

    output = CategoricalOutputLayer(last_layer, data_set_collection.labels_shape, session,
                                    regularizer_weighting=regularizer_coeff,
                                    batch_normalize_input=True)

    # train network to convergence

    output.train_till_convergence(data_set_collection.train, data_set_collection.test, learning_rate=0.0001,
                                  continue_epochs=2)

    # report stats

    train_log_prob, train_acc, train_error = output.evaluation_stats(data_set_collection.train)

    test_log_prob, test_acc, test_error = output.evaluation_stats(data_set_collection.test)

    print("%s,%s,%s,%s,%s,%s,%s,%s\n" % (train_log_prob, train_error, train_acc,
                                         test_log_prob, test_error, test_acc,
                                         str(output.get_resizable_dimension_size_all_layers())
                                         .replace(',', '-'),
                                         output.get_parameters_all_layers()))

    # add x nodes with random values to the trained network

    last_layer.resize(END_SIZE, no_splitting_or_pruning=True)

    # train till convergence

    output.train_till_convergence(data_set_collection.train, data_set_collection.test, learning_rate=0.0001,
                                  continue_epochs=2)

    # report stats

    train_log_prob, train_acc, train_error = output.evaluation_stats(data_set_collection.train)

    test_log_prob, test_acc, test_error = output.evaluation_stats(data_set_collection.test)

    print("%s,%s,%s,%s,%s,%s,%s,%s\n" % (train_log_prob, train_error, train_acc,
                                         test_log_prob, test_error, test_acc,
                                         str(output.get_resizable_dimension_size_all_layers())
                                         .replace(',', '-'),
                                         output.get_parameters_all_layers()))

    # remove new node, don't train till convergence

    last_layer.resize(START_SIZE, no_splitting_or_pruning=True)

    # report stats

    train_log_prob, train_acc, train_error = output.evaluation_stats(data_set_collection.train)

    test_log_prob, test_acc, test_error = output.evaluation_stats(data_set_collection.test)

    print("%s,%s,%s,%s,%s,%s,%s,%s\n" % (train_log_prob, train_error, train_acc,
                                         test_log_prob, test_error, test_acc,
                                         str(output.get_resizable_dimension_size_all_layers())
                                         .replace(',', '-'),
                                         output.get_parameters_all_layers()))

    # contrast above with
