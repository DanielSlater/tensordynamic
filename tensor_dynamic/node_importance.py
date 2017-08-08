import numpy as np

from tensor_dynamic.utils import create_hessian_variable_op, get_first_two_derivatives_op


def node_importance_by_dummy_activation_from_input_layer(layer, data_set_train, data_set_validation):
    shape = (1,) + tuple(int(x) for x in layer.input_placeholder.get_shape()[1:])
    all_pos_1 = np.ones(shape=shape, dtype=np.float32)

    all_zero = np.zeros(shape=shape, dtype=np.float32)

    all_neg_1 = -np.ones(shape=shape, dtype=np.float32)

    importance = layer._session.run(layer.activation_predict,
                                    feed_dict={layer.input_placeholder:
                                                   np.append(np.append(all_pos_1, all_zero, axis=0), all_neg_1,
                                                             axis=0)})

    return np.sum(importance, axis=0)


def node_importance_by_real_activation_from_input_layer(layer, data_set_train, data_set_validation):
    data_set = data_set_train or data_set_validation
    if data_set is not None:
        importance = layer._session.run(layer.activation_predict,
                                        feed_dict={layer.input_placeholder:
                                                       data_set.features})

        return np.sum(importance, axis=0)
    else:
        return node_importance_random(layer, data_set, data_set_validation)


def node_importance_by_real_activation_from_input_layer_variance(layer, data_set_train, data_set_validation):
    data_set = data_set_train or data_set_validation
    if data_set is not None:
        importance = layer._session.run(layer.activation_predict,
                                        feed_dict={layer.input_placeholder:
                                                       data_set.features})
        return np.var(importance, axis=0)
    else:
        return node_importance_random(layer, data_set, data_set_validation)


def node_importance_by_square_sum(layer, data_set_train, data_set_validation):
    data_set = data_set_train or data_set_validation
    # TODO by bound variable
    weights, bias = layer._session.run([layer._weights, layer._bias])

    return np.sum(np.square(weights), axis=0) + np.square(bias)


def node_importance_random(layer, data_set_train, data_set_validation):
    return np.random.normal(size=(layer.get_resizable_dimension_size()))


def node_importance_by_removal(layer, data_set_train, data_set_validation):
    data_set = data_set_train or data_set_validation

    # TODO by bound variable
    if data_set is None:
        return node_importance_random(layer, data_set)

    base_error = layer._session.run(layer.last_layer.target_loss_op_predict,
                                    feed_dict={layer.input_placeholder:
                                                   data_set.features,
                                               layer.target_placeholder:
                                                   data_set.labels})

    weights, bias = layer._session.run([layer._weights, layer._bias])

    errors = []
    for i in range(layer.get_resizable_dimension_size()):
        # null node
        new_bias = np.copy(bias)
        new_bias[i] = 0.

        new_weights = np.copy(weights)
        new_weights[:, i] = 0.
        layer.weights = new_weights
        layer.bias = new_bias

        # layer._session.run([tf.assign(layer._weights, new_weights), tf.assign(layer._bias, new_bias)])
        error_without_node = layer.session.run(layer.last_layer.target_loss_op_predict,
                                               feed_dict={layer.input_placeholder:
                                                              data_set.features,
                                                          layer.target_placeholder:
                                                              data_set.labels})
        errors.append(base_error - error_without_node)

    layer.weights = weights
    layer.bias = bias

    return errors


def node_importance_optimal_brain_damage(layer, data_set_train, data_set_validation):
    data_set = data_set_train or data_set_validation

    if data_set is None:
        return node_importance_random(layer, data_set, data_set_validation)

    weights_hessian_op, bias_hessian_op = layer.hessien_with_respect_to_error_op

    weights, bias, weights_hessian, bias_hessian = layer.session.run(
        [layer._weights, layer._bias, weights_hessian_op, bias_hessian_op],
        feed_dict={layer.input_placeholder: data_set.features,
                   layer.target_placeholder: data_set.labels}
    )

    weights_squared = np.square(weights)
    bias_squared = np.square(bias)

    return np.sum(weights_squared * weights_hessian, axis=0) + bias_squared * bias_hessian


def node_importance_full_taylor_series(layer, data_set_train, data_set_validation):
    data_set = data_set_validation

    if data_set is None:
        return node_importance_random(layer, data_set, data_set_validation)

    weights_jacobean_op, bias_jacobean_op = layer.gradients_with_respect_to_error_op
    weights_hessian_op, bias_hessian_op = layer.hessien_with_respect_to_error_op

    weights, bias, weights_jacobean, bias_jacobean, weights_hessian, bias_hessian = layer.session.run(
        [layer._weights, layer._bias, weights_jacobean_op, bias_jacobean_op, weights_hessian_op, bias_hessian_op],
        feed_dict={layer.input_placeholder: data_set.features,
                   layer.target_placeholder: data_set.labels}
    )

    weights_squared = np.square(weights)
    bias_squared = np.square(bias)

    return np.sum((weights_squared * weights_hessian) * .5 + weights * weights_jacobean,
                  axis=0) + (bias_squared * bias_hessian) * .5 + bias * bias_jacobean


def node_importance_error_derrivative(layer, data_set_train, data_set_validation):
    data_set = data_set_validation

    if data_set is None:
        return node_importance_random(layer, data_set, data_set_validation)

    weights_jacobean_op, bias_jacobean_op = layer.gradients_with_respect_to_error_op

    weights, bias, weights_jacobean, bias_jacobean = layer.session.run(
        [layer._weights, layer._bias, weights_jacobean_op, bias_jacobean_op],
        feed_dict={layer.input_placeholder: data_set.features,
                   layer.target_placeholder: data_set.labels}
    )

    return np.sum(weights * weights_jacobean, axis=0) + bias * bias_jacobean