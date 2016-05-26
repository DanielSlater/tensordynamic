import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant=1.0):
    """ Xavier initialization of network weights
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    Parameters
    ----------
    fan_in : int
        Number of input connections to this matrix

    fan_out : int
        Number of output connections from this matrix

    constant : float32
        Scales the output

    Returns
    -------
        A tensor of the specified shape filled with random uniform values.
    """
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype="float")


def tf_resize(session, tensor, new_dims=None, new_values=None):
    """
    Resize a tensor or variable

    Parameters
    ----------
    session : tensorflow.Session
        The session within which this variable resides
    tensor : tensorflow.Tensor or tensorflow.Variable
        The variable or tensor we wish to resize
    new_dims : Optional [int]
        The dimensions we want the tensor transformed to. If None will be set to the dims of the new_values array
    new_values : Optional numpy.arrray
        If passed then these values are given to the resized tensor
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