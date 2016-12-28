import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def xavier_init(fan_in, fan_out, constant=1.0):
    """ Xavier initialization of network weights
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    Args:
        fan_in (int): Number of input connections to this matrix
        fan_out (int): Number of output connections from this matrix
        constant (float32): Scales the output

    Returns:
        tensorflow.Tensor: A tensor of the specified shape filled with random uniform values.
    """
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype="float")


def tf_resize(session, tensor, new_dims=None, new_values=None):
    """Resize a tensor or variable

    Args:
        session (tensorflow.Session): The session within which this variable resides
        tensor (tensorflow.Tensor or tensorflow.Variable): The variable or tensor we wish to resize
        new_dims ([int]): The dimensions we want the tensor transformed to. If None will be set to the dims of the new_values array
        new_values (numpy.array): If passed then these values are given to the resized tensor
    """
    if new_dims is None and new_values is not None:
        new_dims = new_values.shape

    if new_values is not None:
        if hasattr(new_values, '__call__'):
            new_values = new_values()

        assign = tf.assign(tensor, new_values, validate_shape=False)
        session.run(assign)

    if tuple(tensor.get_shape().as_list()) != new_dims:
        new_shape = tf.python.framework.tensor_shape.TensorShape(new_dims)
        if hasattr(tensor, '_variable'):
            tensor._variable._shape = new_shape
            tensor._snapshot._shape = new_shape
        elif hasattr(tensor, '_shape'):
            tensor._shape = new_shape
        else:
            raise NotImplementedError('unrecognized type %s' % type(tensor))

        for output in tensor.op.outputs:
            output._shape = new_shape


def tf_resize_cascading(session, variable, new_values):
    raise NotImplementedError()
    tf_resize(session, variable, tuple(new_values.shape), new_values)
    consumers = variable._as_graph_element().consumers()
    for consumer in consumers:
        for output in consumer.outputs:
            print output


def train_till_convergence(train_one_epoch_function, continue_epochs=3, max_epochs=10000, log=False,
                           on_no_improvement_func=None):
    """Runs the train_one_epoch_function until we go continue_epochs without improvement in the best error

    Args:
        on_no_improvement_func (()->()): Called whenever we don't see an improvement in training, can be used to change
            the learning rate
        train_one_epoch_function (()->int): Function that when called runs one epoch of training returning the error
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


def get_tf_adam_optimizer_variables(optimizer):
    for slot_values in optimizer._slots.values():
        for value in slot_values.values():
            yield value

    yield optimizer._beta1_power
    yield optimizer._beta2_power
