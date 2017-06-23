import tensorflow as tf

from tensor_dynamic.data.cifar_data import get_cifar_100_data_set_collection
from tensor_dynamic.layers.convolutional_layer import ConvolutionalLayer
from tensor_dynamic.layers.flatten_layer import FlattenLayer
from tensor_dynamic.layers.input_layer import NoisyInputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.layers.max_pool_layer import MaxPoolLayer
from tensor_dynamic.layers.categorical_output_layer import CategoricalOutputLayer

data_set_collection = get_cifar_100_data_set_collection()

with tf.Session() as session:
    non_liniarity = tf.nn.relu

    regularizer_coeff = 0.001
    last_layer = NoisyInputLayer(data_set_collection.features_shape, session)

    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)
    #
    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)
    #
    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)
    #
    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)
    #
    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)
    #
    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)

    last_layer = ConvolutionalLayer(last_layer, (5, 5, 32), stride=(1, 1, 1), session=session,
                                    non_liniarity=non_liniarity)

    last_layer = MaxPoolLayer(last_layer, session=session)

    last_layer = ConvolutionalLayer(last_layer, (5, 5, 64), stride=(1, 1, 1), session=session,
                                    non_liniarity=non_liniarity)

    last_layer = MaxPoolLayer(last_layer, session=session)

    last_layer = FlattenLayer(last_layer, session=session)

    last_layer = Layer(last_layer, 1024, session, non_liniarity=non_liniarity)

    # last_layer = Layer(last_layer, 256, session, non_liniarity=non_liniarity)

    output = CategoricalOutputLayer(last_layer, data_set_collection.labels_shape, session,
                                    regularizer_weighting=regularizer_coeff)

    output.train_till_convergence(data_set_collection.train, data_set_collection.test, learning_rate=0.0001,
                                  continue_epochs=3)


    # (7508.6528, 0.97310001)
    # INFO:tensor_dynamic.layers.output_layer:iterations = 23 error = 7508.65
