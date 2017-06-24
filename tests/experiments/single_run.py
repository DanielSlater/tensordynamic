import tensorflow as tf

from tensor_dynamic.data.cifar_data import get_cifar_100_data_set_collection
from tensor_dynamic.layers.convolutional_layer import ConvolutionalLayer
from tensor_dynamic.layers.flatten_layer import FlattenLayer
from tensor_dynamic.layers.input_layer import NoisyInputLayer, InputLayer
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.layers.max_pool_layer import MaxPoolLayer
from tensor_dynamic.layers.categorical_output_layer import CategoricalOutputLayer

data_set_collection = get_cifar_100_data_set_collection()

with tf.Session() as session:
    non_liniarity = tf.nn.relu

    regularizer_coeff = 0.0
    last_layer = NoisyInputLayer(data_set_collection.features_shape, session, noise_std=1.)

    last_layer = FlattenLayer(last_layer, session)

    for _ in range(4):
        last_layer = HiddenLayer(last_layer, 300, session, non_liniarity=non_liniarity, input_noise_std=1.)

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

    # output.train_till_convergence(data_set_collection.train, data_set_collection.test, learning_rate=0.00001,
    #                               continue_epochs=2)

    output.learn_structure_random(data_set_collection.train, data_set_collection.test)

    train_log_prob, train_error, train_acc = output.evaluation_stats(data_set_collection.train)

    test_log_prob, test_error, test_acc = output.evaluation_stats(data_set_collection.test)

    print("%s,%s,%s,%s,%s,%s,%s,%s\n" % (train_log_prob, train_error, train_acc,
                                         test_log_prob, test_error, test_acc,
                                         str(output.get_resizable_dimension_size_all_layers())
                                         .replace(',', '-'),
                                         output.get_parameters_all_layers()))

    # (7508.6528, 0.97310001)
    # INFO:tensor_dynamic.layers.output_layer:iterations = 23 error = 7508.65
