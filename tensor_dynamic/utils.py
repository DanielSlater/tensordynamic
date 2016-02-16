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