import functools

import tensorflow as tf

from tensor_dynamic.bayesian_resizing_net import create_flat_network
from tests.base_tf_testcase import get_mnist_data


def do_grid_search(data_set_collection, model_functions, file_name, learning_rate=0.001,
                   continue_epochs=4, **extra_parameters):
    """Run a grid search and write all results to csv file

    Args:
        continue_epochs (int):
        data_set_collection (tensor_dynamic.data.data_set_collection.DataSetCollection):
        model_functions: Function that returns an iterator of functions that when given a tensorflow session create the
            model we want to run for each element in the search
        extra_parameters (dict):
        learning_rate (float):
    """
    write_parameters_file(data_set_collection, extra_parameters, file_name, learning_rate, continue_epochs)
    with open(file_name, 'w') as result_file:
        result_file.write(
            'log_prob_train, error_train, accuracy_train, error_test, accuracy_test, log_prob_test, dimensions, parameters\n')

        for model_function in model_functions(data_set_collection):
            with tf.Session() as session:
                model = model_function(session)
                model.train_till_convergence(data_set_collection.train, data_set_collection.test,
                                             learning_rate=learning_rate, continue_epochs=continue_epochs,
                                             optimizer=extra_parameters['optimizer'])

                train_log_prob, train_error, train_acc = model.evaluation_stats(data_set_collection.train.features,
                                                                                data_set_collection.train.labels)

                test_log_prob, test_error, test_acc = model.evaluation_stats(data_set_collection.test.features,
                                                                             data_set_collection.test.labels)

                result_file.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (train_log_prob, train_error, train_acc,
                                                                 test_log_prob, test_error, test_acc,
                                                                 str(model.get_resizable_dimension_size_all_layers())
                                                                 .replace(',', '-'),
                                                                 model.get_parameters_all_layers()))


def write_parameters_file(data_set_collection, extra_parameters, file_name, learning_rate, continue_epochs):
    with open(file_name + '.txt', 'w') as param_file:
        param_file.write("data_set=%s\n" % (data_set_collection.name,))
        param_file.write("data_set_normalized=%s\n" % (data_set_collection.normlized,))
        param_file.write("learning_rate=%s\n" % (learning_rate,))
        param_file.write("continue_epochs=%s\n" % (continue_epochs,))
        for key, value in extra_parameters.iteritems():
            param_file.write("%s=%s\n" % (key, value))


def flat_model_functions(data_set_collection, regularizer, activation_func, use_noisy_input_layer):
    def get_model(session, parameters):
        dims = data_set_collection.features_shape + parameters + data_set_collection.labels_shape
        return create_flat_network(dims, session, regularizer_coeff=regularizer, activation_func=activation_func,
                                   use_noisy_input_layer=use_noisy_input_layer)

    # 1 layer
    for layer_1 in [10, 20, 30, 50, 100, 200]:
        yield functools.partial(get_model, parameters=(layer_1,))

        for layer_2 in [10, 20, 30, 50, 100]:
            if layer_2 <= layer_1:
                yield functools.partial(get_model, parameters=(layer_1, layer_2))

                for layer_3 in [10, 20, 30, 50, 100]:

                    if layer_3 <= layer_2:
                        yield functools.partial(get_model, parameters=(layer_1, layer_2, layer_3))


if __name__ == '__main__':
    regularizer = 0.0

    data_set_collection = get_mnist_data()
    use_noisy_input_layer = True
    do_grid_search(data_set_collection,
                   functools.partial(flat_model_functions,
                                     regularizer=regularizer,
                                     activation_func=tf.nn.relu,
                                     use_noisy_input_layer=use_noisy_input_layer),
                   'flat_mnist_grid_search_no_reg.csv',
                   regularizer=regularizer,
                   use_noisy_input_layer=use_noisy_input_layer,
                   activation_func='relu',
                   network='flat',
                   optimizer=tf.train.AdamOptimizer)
