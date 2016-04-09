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


def tf_resize(session, tensor, new_dims, new_values=None):
    if new_values is not None:
        if hasattr(new_values, '__call__'):
            new_values = new_values()

        assign = tf.assign(tensor, new_values, validate_shape=False)
        session.run(assign)

    if tuple(tensor.get_shape().as_list()) != new_dims:
        if hasattr(tensor, '_variable'):
            tensor._variable._shape = tf.python.framework.tensor_shape.TensorShape(new_dims)
        elif hasattr(tensor, '_shape'):
            tensor._shape = tf.python.framework.tensor_shape.TensorShape(new_dims)
        else:
            raise NotImplementedError('unrecognized type %s' % type(tensor))