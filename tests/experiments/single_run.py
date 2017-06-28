import tensorflow as tf

from tensor_dynamic.data.cifar_data import get_cifar_100_data_set_collection
from tensor_dynamic.data.mnist_data import get_mnist_data_set_collection
from tensor_dynamic.layers.convolutional_layer import ConvolutionalLayer
from tensor_dynamic.layers.flatten_layer import FlattenLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.layers.max_pool_layer import MaxPoolLayer
from tensor_dynamic.layers.categorical_output_layer import CategoricalOutputLayer

# data_set_collection = get_cifar_100_data_set_collection()
data_set_collection = get_mnist_data_set_collection()

with tf.Session() as session:
    non_liniarity = tf.nn.relu

    regularizer_coeff = 0.001
    last_layer = InputLayer(data_set_collection.features_shape,
                            # drop_out_prob=.5,
                            # layer_noise_std=1.
                            )

    # last_layer = FlattenLayer(last_layer, session)

    for _ in range(1):
        last_layer = HiddenLayer(last_layer, 100, session, non_liniarity=non_liniarity,
                                 batch_normalize_input=True)

    # last_layer = ConvolutionalLayer(last_layer, (5, 5, 32), stride=(1, 1, 1), session=session,
    #                                 non_liniarity=non_liniarity)
    #
    # last_layer = MaxPoolLayer(last_layer, session=session)
    #
    # last_layer = ConvolutionalLayer(last_layer, (5, 5, 64), stride=(1, 1, 1), session=session,
    #                                 non_liniarity=non_liniarity)
    #
    # last_layer = MaxPoolLayer(last_layer, session=session)
    #
    # last_layer = FlattenLayer(last_layer, session=session)
    #
    # last_layer = HiddenLayer(last_layer, 1024, session, non_liniarity=non_liniarity)
    #
    # last_layer = HiddenLayer(last_layer, 512, session, non_liniarity=non_liniarity)
    #
    # last_layer = HiddenLayer(last_layer, 512, session, non_liniarity=non_liniarity)

    output = CategoricalOutputLayer(last_layer, data_set_collection.labels_shape, session,
                                    regularizer_weighting=regularizer_coeff)

    output.train_till_convergence(data_set_collection.train, data_set_collection.test, learning_rate=0.00001,
                                  continue_epochs=2)

    # output.learn_structure_layer_by_layer(data_set_collection.train, data_set_collection.test,
    #                                       start_learn_rate=0.0001, continue_learn_rate=0.00001)

    train_log_prob, train_acc, train_error = output.evaluation_stats(data_set_collection.train)

    test_log_prob, test_acc, test_error = output.evaluation_stats(data_set_collection.test)

    print("%s,%s,%s,%s,%s,%s,%s,%s\n" % (train_log_prob, train_error, train_acc,
                                         test_log_prob, test_error, test_acc,
                                         str(output.get_resizable_dimension_size_all_layers())
                                         .replace(',', '-'),
                                         output.get_parameters_all_layers()))

    # (7508.6528, 0.97310001)
    # INFO:tensor_dynamic.layers.output_layer:iterations = 23 error = 7508.65
