import math

import numpy as np
import tensorflow as tf


def noise_weight_extender(array, extended_dimensions, mean=0.0, var=None):
    """
    Extends a numpy array to have a new dimension, new values are filled in using random gaussian noise

    Parameters
    ----------
    array : np.array
        The array we want to resize
    extended_dimensions : [int]
        The size to extend the array to, must be larger than the current array
    mean : float
    var : float

    Returns
    -------
    np.array

    Array will be of size new_dims
    """
    new_values = array

    if len(extended_dimensions) == 1:
        if extended_dimensions[0] > array.shape[-1]:
            new_values = np.append(array, np.zeros([extended_dimensions[0] - array.shape[-1]])) \
                .astype(array.dtype)
    else:
        if extended_dimensions[0] > array.shape[0]:
            new_values = np.append(new_values,
                                   np.random.normal(scale=var or (1.0 / math.sqrt(float(extended_dimensions[0]))),
                                                    loc=mean,
                                                    size=(
                                                    extended_dimensions[0] - new_values.shape[0], new_values.shape[1]))
                                   .astype(array.dtype),
                                   axis=0)
        if extended_dimensions[1] > array.shape[1]:
            new_values = np.append(new_values,
                                   np.random.normal(
                                       scale=var or (1.0 / math.sqrt(float(extended_dimensions[1])) / 100.),
                                       loc=mean,
                                       size=(new_values.shape[0], extended_dimensions[1] - new_values.shape[1]))
                                   .astype(array.dtype),
                                   axis=1)

    return new_values
