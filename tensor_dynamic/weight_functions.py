import math

import numpy as np
import tensorflow as tf


def noise_weight_extender(array, extended_dimensions, mean=0.0, var=None):
    """Extends a numpy array to have a new dimension, new values are filled in using random gaussian noise

    Args:
        array (np.array): The array we want to resize
        extended_dimensions ([int]): The size to extend the array to, must be larger than the current array
        mean (float):
        var (float):

    Returns:
        np.array : Array will be of size new_dims
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
                                                        extended_dimensions[0] - new_values.shape[0],
                                                        new_values.shape[1]))
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


def array_extend(array, vectors_to_extend, noise_std=None, halve_extended_vectors=False):
    """Extends the array arg by the column/row specified in vectors_to_extend duplicated

    Examples:
        a = np.array([[0, 1, 0],
                      [0, 1, 0]])
        array_split_extension(a, {1: [1]}) # {1: [1]} means duplicate column, with index 1
        # np.array([[0, 1, 0, 1], [0, 1, 0, 1]]))

    Args:
        array (np.array): The array we want to split
        vectors_to_extend ({int:[int]): The keys are the axis we want to split, 0 = rows, 1 = keys,
            while the values are which rows/columns along that axis we want to duplicate
        noise_std (float): If set some random noise is applied to the extended column and subtracted from the
            duplicated column. The std of the noise is the value of this column
        halve_extended_vectors (bool): If True then extended vector and vector copied from both halved so as to leave
            the network activation, relatively unchanged

    Returns:
        np.array : The array passed in as array arg but now extended
    """
    for axis, split_indexes in vectors_to_extend.iteritems():
        for x in split_indexes:
            split_args = [slice(None)] * array.ndim
            split_args[axis] = x
            add_weights = np.copy(array[split_args])
            reshape_args = list(array.shape)
            reshape_args[axis] = 1
            add_weights = add_weights.reshape(reshape_args)

            if halve_extended_vectors:
                add_weights *= .5
                array[split_args] *= .5

            if noise_std:
                random_noise = np.random.normal(scale=noise_std, size=add_weights.shape)
                add_weights += random_noise
                array[split_args] -= random_noise

            array = np.r_[str(axis), array, add_weights]
    return array
