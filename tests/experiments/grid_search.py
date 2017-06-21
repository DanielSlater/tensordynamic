import functools

import tensorflow as tf

from tensor_dynamic.bayesian_resizing_net import create_flat_network
from tests.base_tf_testcase import get_mnist_data


def do_grid_search(data_set_collection, model_functions, extra_parameters={}, learning_rate=0.001):
    with open('grid_search.csv.txt', 'w') as param_file:
        param_file.write("learning_rate=%s\n" % (learning_rate,))
        param_file.write(str(extra_parameters))
    with open('grid_search.csv', 'w') as result_file:
        result_file.write('error_train, accuracy_train, error_test, accuracy_test, dimensions, parameters\n')

        for model_function in model_functions(data_set_collection):
            with tf.Session() as session:
                model = model_function(session)
                model.train_till_convergence(data_set_collection.train, data_set_collection.test,
                                             learning_rate=learning_rate)

                train_error, train_acc = session.run([model.loss, model._accuracy],
                                                     feed_dict={
                                                         model.input_placeholder: data_set_collection.test.features,
                                                         model.target_placeholder: data_set_collection.test.labels})

                test_error, test_acc = session.run([model.loss, model._accuracy],
                                                   feed_dict={
                                                       model.input_placeholder: data_set_collection.test.features,
                                                       model.target_placeholder: data_set_collection.test.labels})

                result_file.write("%s,%s,%s,%s,%s,%s\n" % (train_error, train_acc, test_error, test_acc,
                                                           str(model.get_resizable_dimension_size_all_layers())
                                                           .replace(',', '-'),
                                                           model.get_parameters_all_layers()))


def flat_model_functions(data_set_collection, regularizer):
    def get_model(session, parameters):
        dims = data_set_collection.features_shape + parameters + data_set_collection.labels_shape
        return create_flat_network(dims, session, regularizer_coeff=regularizer)

    # 1 layer
    for layer_1 in range(20, 200, 10):
        yield functools.partial(get_model, parameters=(layer_1,))

        for layer_2 in range(20, 150, 10):
            if layer_2 <= layer_1:
                yield functools.partial(get_model, parameters=(layer_1, layer_2))

                for layer_3 in range(20, 150, 10):

                    if layer_3 <= layer_2:
                        yield functools.partial(get_model, parameters=(layer_1, layer_2, layer_3))


if __name__ == '__main__':
    regularizer = 0.001

    data_set_collection = get_mnist_data()
    do_grid_search(data_set_collection,
                   functools.partial(flat_model_functions, regularizer=regularizer),
                   extra_parameters={'regularizer': regularizer,
                                     'activation_func': 'sigmoid'})