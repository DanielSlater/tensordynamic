import logging

import itertools
import numpy as np
import tensorflow as tf
from collections import Iterable
from tensorflow.python.framework.tensor_shape import TensorShape

logger = logging.getLogger(__name__)


def xavier_init(fan_in, fan_out, constant=1.0):
    """ Xavier initialization of network weights
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    Args:
        fan_in (int | tuple of ints): Number of input connections to this matrix
        fan_out (int | tuple of ints): Number of output connections from this matrix
        constant (float32): Scales the output

    Returns:
        tensorflow.Tensor: A tensor of the specified shape filled with random uniform values.
    """
    if isinstance(fan_in, Iterable):
        fan_in = get_product_of_iterable(fan_in)
    if isinstance(fan_out, Iterable):
        fan_out = get_product_of_iterable(fan_out)

    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def weight_init(shape, constant=1.0):
    fan_in = get_product_of_iterable(shape[:-1])
    fan_out = get_product_of_iterable(shape[-1:])

    low = -constant * np.sqrt(1.0 / (fan_in + fan_out))
    high = constant * np.sqrt(1.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def bias_init(shape, constant=0.01):
    if isinstance(shape, int):
        shape = (shape,)
    return tf.constant(constant, shape=shape, dtype=tf.float32)


def get_product_of_iterable(iterable):
    """Product of the items in the input e.g. [1,2,3,4] => 24

    Args:
        iterable (iterable of ints):

    Returns:
        int
    """
    product = 1
    for x in iterable:
        product *= x
    return product


def tf_resize(session, tensor, new_dimensions=None, new_values=None, assign_function=None):
    """Resize a tensor or variable

    Args:
        assign_function (tensorflow.Operation): Operation for assigning this variable, this is to stop the graph
            getting overloaded
        session (tensorflow.Session): The session within which this variable resides
        tensor (tensorflow.Tensor or tensorflow.Variable): The variable or tensor we wish to resize
        new_dimensions ([int]): The dimensions we want the tensor transformed to. If None will be set to the dims of the new_values array
        new_values (numpy.array): If passed then these values are given to the resized tensor
    """
    if new_values is not None and new_dimensions is not None:
        if tuple(new_dimensions) != new_values.shape:
            raise ValueError("new_dimsensions and new_values, if set, must have the same shape")

    if new_dimensions is None and new_values is not None:
        new_dimensions = new_values.shape

    if new_values is not None:
        if hasattr(new_values, '__call__'):
            new_values = new_values()

        if assign_function is None:
            assign = tf.assign(tensor, new_values, validate_shape=False)
            session.run(assign)
        else:
            assign_function(new_values)
    elif isinstance(tensor, tf.Variable):
        current_vals = session.run(tensor)
        new_values = np.resize(current_vals, new_dimensions)

        if assign_function is None:
            assign = tf.assign(tensor, new_values, validate_shape=False)
            session.run(assign)
        else:
            assign_function(new_values)

    if tuple(tensor.get_shape().as_list()) != new_dimensions:
        new_shape = TensorShape(new_dimensions)
        if hasattr(tensor, '_variable'):
            for i in range(len(new_dimensions)):
                tensor._variable._shape._dims[i]._value = new_dimensions[i]
                tensor._snapshot._shape._dims[i]._value = new_dimensions[i]
                tensor._initial_value._shape._dims[i]._value = new_dimensions[i]

            tensor._snapshot._shape = new_shape
            tensor._variable._shape = new_shape
            tensor._initial_value._shape = new_shape
        elif hasattr(tensor, '_shape'):
            for i in range(len(new_dimensions)):
                tensor._shape._dims[i]._value = new_dimensions[i]
        else:
            raise NotImplementedError('unrecognized type %s' % type(tensor))

        for output in tensor.op.outputs:
            output._shape = new_shape

        for input in tensor.op.inputs:
            if len(input._shape) == len(new_shape):
                input._shape = new_shape
            elif len(input._shape) == len(new_shape) + 1:
                input._shape = TensorShape((input._shape[0]._value, ) + new_dimensions)
            elif len(input._shape) == 0:
                pass
            elif len(input._shape) +1 == len(new_shape) and new_shape[0]._value is None:
                input._shape = TensorShape(new_dimensions[1:])
            else:
                raise Exception("could not deal with this input")


def tf_resize_cascading(session, variable, new_values):
    #raise NotImplementedError()
    tf_resize(session, variable, tuple(new_values.shape), new_values)
    consumers = variable._as_graph_element().consumers()
    for consumer in consumers:
        for output in consumer.outputs:
            print output


def train_till_convergence(train_one_epoch_function, continue_epochs=3, max_epochs=10000,
                           log=False,
                           on_no_improvement_func=None):
    """Runs the train_one_epoch_function until we go continue_epochs without improvement in the best error

    Args:
        on_no_improvement_func (()->()): Called whenever we don't see an improvement in training, can be used to change
            the learning rate
        train_one_epoch_function (()->number): Function that when called runs one epoch of training returning the error
            from training.
        continue_epochs (int): The number of epochs without improvement before we terminate training, default 3
        max_epochs (int): The max number of epochs we can run for. default 10000
        log (bool): If true print result of each epoch

    Returns:
        int: The error we got for the final training epoch
    """
    best_error = train_one_epoch_function()
    error = best_error
    epochs_since_best_error = 0

    for epochs in xrange(1, max_epochs):
        error = train_one_epoch_function()
        if log:
            logger.info("epochs %s error %s", epochs, error)

        if error < best_error:
            best_error = error
            epochs_since_best_error = 0
        else:
            epochs_since_best_error += 1
            if epochs_since_best_error >= continue_epochs:
                if log:
                    logger.info("finished with best error %s", best_error)
                break

            if on_no_improvement_func:
                on_no_improvement_func()

    return error


def get_tf_optimizer_variables(optimizer):
    """Get all the tensorflow variables in an optimzier, for use in initialization

    Args:
        optimizer (tf.train.Optimizer): Some kind of tensorflow optimizer

    Returns:
        Iterable of tf.Variable
    """
    if isinstance(optimizer, tf.train.AdamOptimizer):
        for var in _get_optimzer_slot_variables(optimizer):
            yield var
        yield optimizer._beta1_power
        yield optimizer._beta2_power
    elif isinstance(optimizer, tf.train.RMSPropOptimizer):
        for var in _get_optimzer_slot_variables(optimizer):
            yield var
    elif isinstance(optimizer, tf.train.GradientDescentOptimizer):
        pass
    else:
        raise TypeError("Unsupported optimizer %s" % (type(optimizer),))


def _get_optimzer_slot_variables(optimizer):
    count = 0
    for slot_values in optimizer._slots.values():
        for value in slot_values.values():
            count += 1
            yield value

    if count == 0:
        raise Exception("Found no variables in optimizer, you may need to call minimize on this optimizer before calling this method")


def _iterate_coords(tensor):
    if len(tensor.get_shape()) == 1:
        for i in range(tensor.get_shape()[0]):
            yield (i,), (1,)
    else:
        for i in range(tensor.get_shape()[0]):
            for j in range(tensor.get_shape()[1]):
                yield (i, j), (1, 1)


def _variable_size(variable):
    size = 1
    for dim in variable.get_shape():
        size *= int(dim)
    return size


def create_hessian_op(tensor_op, variables, session):
    mat = []
    for v1 in variables:
        for v2 in variables:
            temp = []
            # computing derivative twice, first w.r.t v2 and then w.r.t v1
            first_derivative = tf.gradients(tensor_op, v2)[0]
            for begin, size in _iterate_coords(v2):
                temp.append(tf.gradients(tf.slice(first_derivative, begin=begin, size=size), v1)[0])
            # tensorflow returns None when there is no gradient, so we replace None with, maybe we should just fail...
            # temp = [0. if t is None else t for t in temp]

            derivatives = tf.concat(0, temp)

        mat.append(temp)

    raise NotImplementedError()

    return mat


def get_first_two_derivatives_op(loss_op, tensor):
    """Given a loss function get the 2nd derivatives of all variables with respect to the loss function

    Args:
        loss_op:
        tensor:

    Returns:

    """
    # computing derivative twice, first w.r.t v2 and then w.r.t v1
    first_derivative = tf.gradients(loss_op, tensor)[0]
    second_derivative = tf.gradients(first_derivative, tensor)[0]

    return first_derivative, second_derivative


def create_hessian_variable_op(loss_op, tensor):
    """Given a loss function get the 2nd derivatives of all variables with respect to the loss function

    Args:
        loss_op:
        tensor:

    Returns:

    """
    # computing derivative twice, first w.r.t v2 and then w.r.t v1
    first_derivative = tf.gradients(loss_op, tensor)[0]
    second_derivative = tf.gradients(first_derivative, tensor)[0]

    return second_derivative