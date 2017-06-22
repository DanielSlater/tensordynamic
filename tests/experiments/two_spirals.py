import tensorflow as tf

from tensor_dynamic.data.cifar_data import get_cifar_100_data_set_collection
from tensor_dynamic.data.two_spirals import get_two_spirals_data_set_collection
from tensor_dynamic.layers.convolutional_layer import ConvolutionalLayer
from tensor_dynamic.layers.flatten_layer import FlattenLayer
from tensor_dynamic.layers.input_layer import NoisyInputLayer, InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.layers.max_pool_layer import MaxPoolLayer
from tensor_dynamic.layers.output_layer import CategoricalOutputLayer, OutputLayer, BinaryLayer

data_set_collection = get_two_spirals_data_set_collection()

with tf.Session() as session:
    non_liniarity = tf.nn.tanh

    regularizer_coeff = 0.001
    last_layer = InputLayer(data_set_collection.features_shape, session)

    last_layer = Layer(last_layer, 5, session, non_liniarity=non_liniarity)

    last_layer = Layer(last_layer, 5, session, non_liniarity=non_liniarity)

    last_layer = Layer(last_layer, 5, session, non_liniarity=non_liniarity)
    #
    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)
    #
    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)
    #
    # last_layer = Layer(last_layer, 300, session, non_liniarity=non_liniarity)

    output = BinaryLayer(last_layer, session, regularizer_weighting=regularizer_coeff)

    output.train_till_convergence(data_set_collection.train, data_set_collection.test, learning_rate=0.0001,
                                  continue_epochs=3)


    # (7508.6528, 0.97310001)
    # INFO:tensor_dynamic.layers.output_layer:iterations = 23 error = 7508.65
