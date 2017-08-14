from collections import defaultdict

import tensorflow as tf
from tensor_dynamic.data.cifar_data import get_cifar_100_data_set_collection
from tensor_dynamic.data.mnist_data import get_mnist_data_set_collection
from tensor_dynamic.layers.categorical_output_layer import CategoricalOutputLayer
from tensor_dynamic.layers.flatten_layer import FlattenLayer
from tensor_dynamic.layers.hidden_layer import node_importance_by_square_sum, HiddenLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.node_importance import node_importance_by_dummy_activation_from_input_layer, node_importance_random, \
    node_importance_optimal_brain_damage, node_importance_full_taylor_series, \
    node_importance_by_real_activation_from_input_layer_variance, node_importance_by_removal, \
    node_importance_error_derrivative
from tensor_dynamic.node_importance import node_importance_by_real_activation_from_input_layer

NUM_TRIES = 15


def dummy_random_weights():
    raise Exception()

start = 400
end = 380


def main(file_name_all="pruning_tests%s-%s-%s.csv" % ('_noise=.5', start, end), file_name_avg="pruning_tests%s-%s-%s.csv" % ('_noise=.5', start, end)):
    data_set_collections = [get_mnist_data_set_collection(validation_ratio=.15),
                            get_cifar_100_data_set_collection(validation_ratio=.15)]
    methods = [node_importance_by_dummy_activation_from_input_layer,
               node_importance_by_real_activation_from_input_layer,
               node_importance_by_square_sum,
               node_importance_by_removal,
               node_importance_random,
               node_importance_optimal_brain_damage,
               node_importance_full_taylor_series,
               node_importance_by_real_activation_from_input_layer_variance,
               node_importance_error_derrivative,
               dummy_random_weights
               ]

    final_dict = defaultdict(lambda: [])

    with open(file_name_all, 'w') as result_file:
        result_file.write(
            'method, data_set, before_prune_train, before_prune_validation, before_prune_trest, after_prune_train, after_prune_validataion, after_prune_test, after_converge_train, after_converge_validataion, after_converge_test, converge_iterations\n')
        for data in data_set_collections:
            for _ in range(NUM_TRIES):
                tf.reset_default_graph()
                with tf.Session() as session:
                    input_layer = InputLayer(data.features_shape)

                    if len(data.features_shape) > 1:
                        input_layer = FlattenLayer(input_layer)

                    layer = HiddenLayer(input_layer, start, session=session,
                                        layer_noise_std=.5,
                                        node_importance_func=None,
                                        non_liniarity=tf.nn.relu,
                                        batch_normalize_input=True)
                    output = CategoricalOutputLayer(layer, data.labels_shape,
                                                    batch_normalize_input=True,
                                                    regularizer_weighting=0.01,
                                                    layer_noise_std=.5
                                                    )

                    output.train_till_convergence(data.train, data.validation, learning_rate=0.0001)

                    state = output.get_network_state()

                    for method in methods:
                        output.set_network_state(state)
                        layer._node_importance_func = method

                        _, _, target_loss_test_before_resize_test = output.evaluation_stats(data.test)
                        _, _, target_loss_test_before_resize_validation = output.evaluation_stats(data.validation)
                        _, _, target_loss_test_before_resize_train = output.evaluation_stats(data.train)

                        no_splitting_or_pruning = method == dummy_random_weights

                        layer.resize(end, data_set_train=data.train,
                                     data_set_validation=data.validation,
                                     no_splitting_or_pruning=no_splitting_or_pruning)

                        _, _, target_loss_test_after_resize_test = output.evaluation_stats(data.test)
                        _, _, target_loss_test_after_resize_validation = output.evaluation_stats(data.validation)
                        _, _, target_loss_test_after_resize_train = output.evaluation_stats(data.train)

                        error, iterations = output.train_till_convergence(data.train, data.validation,
                                                                          learning_rate=0.0001)

                        _, _, after_converge_test = output.evaluation_stats(data.test)
                        _, _, after_converge_validation = output.evaluation_stats(data.validation)
                        _, _, after_converge_train = output.evaluation_stats(data.train)

                        final_dict[method.__name__].append((target_loss_test_before_resize_train,
                                                            target_loss_test_before_resize_validation,
                                                            target_loss_test_before_resize_test,
                                                            target_loss_test_after_resize_train,
                                                            target_loss_test_after_resize_validation,
                                                            target_loss_test_after_resize_test,
                                                            after_converge_train,
                                                            after_converge_validation,
                                                            after_converge_test))

                        result_file.write('%s,%s,%s,%s,%s,%s,%s,%s, %s, %s, %s, %s\n' % (
                            method.__name__, data.name, target_loss_test_before_resize_train,
                            target_loss_test_before_resize_validation,
                            target_loss_test_before_resize_test,
                            target_loss_test_after_resize_train,
                            target_loss_test_after_resize_validation,
                            target_loss_test_after_resize_test,
                            after_converge_train,
                            after_converge_validation,
                            after_converge_test,
                            iterations))
                        result_file.flush()

    with open(file_name_avg, "w") as file_avg:
        file_avg.write(
            'method, before_prune_train, before_prune_validataion, before_prune_trest, after_prune_train, after_prune_validataion, after_prune_test, after_converge_train, after_converge_validataion, after_convert_test, test_diff\n')
        for name, values in final_dict.iteritems():
            v_len = float(len(values))
            averages = tuple(sum(x[i] for x in values) / v_len for i in range(len(values[0])))
            averages = averages + (averages[2] - averages[-2],)
            file_avg.write('%s,%s,%s,%s,%s,%s,%s,%s, %s, %s, %s\n' % ((name,) + averages))


if __name__ == '__main__':
    main()
