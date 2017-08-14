import tensorflow as tf

from tensor_dynamic.data.cifar_data import get_cifar_100_data_set_collection
from tensor_dynamic.layers.categorical_output_layer import CategoricalOutputLayer
from tensor_dynamic.layers.flatten_layer import FlattenLayer
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.layers.input_layer import InputLayer

data_set_collection = get_cifar_100_data_set_collection(validation_ratio=.15)
ITERATIONS = 10


def print_stats(data_set_collection, model, layer_num):
    train_log_prob, train_acc, train_error = model.evaluation_stats(data_set_collection.train)
    val_log_prob, val_acc, val_error = model.evaluation_stats(data_set_collection.validation)
    test_log_prob, test_acc, test_error = model.evaluation_stats(data_set_collection.test)
    text = "%s,%s,%s%s,%s,%s,%s,%s,%s,%s,%s, %s\n" % (train_log_prob, train_error, train_acc,
                                                 val_log_prob, val_error, val_acc,
                                                 test_log_prob, test_error, test_acc,
                                                 str(model.get_resizable_dimension_size_all_layers())
                                                 .replace(',', '-'),
                                                 model.get_parameters_all_layers(), layer_num)
    print(text)
    with open('adding_layers.csv', "w") as file_avg:
        file_avg.write(text)


def try_intermediate_layer(layer_num):
    print "add layer at pos " + str(layer_num)
    list(output.all_connected_layers)[layer_num].add_intermediate_layer(
        lambda x: HiddenLayer(x, nodes_per_layer, session,
                              non_liniarity=non_liniarity,
                              batch_normalize_input=True))
    output.train_till_convergence(data_set_collection.train, data_set_collection.validation,
                                  learning_rate=0.0001)
    output.save_checkpoints('cifar-100-layers')
    print_stats(data_set_collection, output, layer_num)
    output.set_network_state(state)


for _ in range(ITERATIONS):
    with tf.Session() as session:
        non_liniarity = tf.nn.relu
        nodes_per_layer = 400

        regularizer_coeff = 0.01
        last_layer = InputLayer(data_set_collection.features_shape,
                                # drop_out_prob=.5,
                                # layer_noise_std=1.
                                )

        last_layer = FlattenLayer(last_layer, session)

        for _ in range(3):
            last_layer = HiddenLayer(last_layer, nodes_per_layer, session, non_liniarity=non_liniarity,
                                     batch_normalize_input=True)

        output = CategoricalOutputLayer(last_layer, data_set_collection.labels_shape, session,
                                        batch_normalize_input=True,
                                        loss_cross_entropy_or_log_prob=True,
                                        regularizer_weighting=regularizer_coeff)

        output.train_till_convergence(data_set_collection.train, data_set_collection.validation,
                                      learning_rate=0.0001)

        state = output.get_network_state()

        output.save_checkpoints('cifar-100-layers')

        print_stats(data_set_collection, output, -1)

        for i in range(3):
            try_intermediate_layer(4 - i)

            # (7508.6528, 0.97310001)
            # INFO:tensor_dynamic.layers.output_layer:iterations = 23 error = 7508.65
