import math

import numpy as np
import tensorflow as tf


def noise_weight_extender(array, new_dimensions, mean=0.0, var=None):
    """Extends a numpy array to have a new dimension, new values are filled in using random gaussian noise

    Args:
        array (np.array): The array we want to resize
        new_dimensions ([int]): The size to extend the array to, must be larger than the current array
        mean (float):
        var (float):

    Returns:
        np.array : Array will be of size new_dims
    """
    if any(x <= 0 for x in new_dimensions):
        raise ValueError("new_dimensions must all be greater than 0 was %s" % (new_dimensions,))
    new_values = array

    if len(new_dimensions) == 1:
        if new_dimensions[0] > array.shape[-1]:
            new_values = np.append(array, np.zeros([new_dimensions[0] - array.shape[-1]])) \
                .astype(array.dtype)
        elif new_dimensions[0] < array.shape[-1]:
            # TODO: smarter downsizing
            new_values = np.resize(new_values, new_dimensions)
    else:
        if new_dimensions[0] > array.shape[0]:
            new_values = np.append(new_values,
                                   np.random.normal(scale=var or (1.0 / math.sqrt(float(new_dimensions[0]))),
                                                    loc=mean,
                                                    size=(
                                                        new_dimensions[0] - new_values.shape[0],
                                                        new_values.shape[1]))
                                   .astype(array.dtype),
                                   axis=0)
        elif new_dimensions[0] < new_values.shape[0]:
            # TODO: smarter downsizing
            new_values = np.resize(new_values, new_dimensions)
        if new_dimensions[1] > array.shape[1]:
            new_values = np.append(new_values,
                                   np.random.normal(
                                       scale=var or (1.0 / math.sqrt(float(new_dimensions[1])) / 100.),
                                       loc=mean,
                                       size=(new_values.shape[0], new_dimensions[1] - new_values.shape[1]))
                                   .astype(array.dtype),
                                   axis=1)
        while new_dimensions[1] < new_values.shape[1]:
            # TODO: smarter downsizing
            new_values = np.delete(new_values, -1, 1)

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


def net_2_deeper_net(bias, noise_std=0.1):
    """
    This is a similar idea to net 2 deeper net from http://arxiv.org/pdf/1511.05641.pdf
    Assumes that this is a linear layer that is being extended and also adds some noise

    Args:
        bias (numpy.array): The bias for the layer we are adding after
        noise_std (Optional float): The amount of normal noise to add to the layer.
            If None then no noise is added
            Default is 0.1
    Returns:
        (numpy.matrix, numpy.array)
        The first item is the weights for the new layer
        Second item is the bias for the new layer
    """
    new_weights = np.matrix(np.eye(bias.shape[0], dtype=bias.dtype))
    new_bias = np.zeros(bias.shape, dtype=bias.dtype)

    if noise_std:
        new_weights = new_weights + np.random.normal(scale=noise_std, size=new_weights.shape)
        new_bias = new_bias + np.random.normal(scale=noise_std, size=new_bias.shape)

    return new_weights, new_bias
