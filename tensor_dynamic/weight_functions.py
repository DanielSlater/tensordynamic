import math

import numpy as np
import tensorflow as tf


def noise_weight_extender(array, new_dimensions, mean=0.0, var=None):
    """Extends a numpy array to have a new dimension, new values are filled in using random gaussian noise

    Args:
        array (np.array): The array we want to resize
        new_dimensions ([int]): The size to extend the array to, must be larger than the current array
        mean (float):
        var (float): How much random noise to add when changing layer size

    Returns:
        np.array : Array will be of size new_dims
    """
    assert len(array.shape) == len(new_dimensions)

    if any(x <= 0 for x in new_dimensions):
        raise ValueError("new_dimensions must all be greater than 0 was %s" % (new_dimensions,))
    new_values = array

    for index in range(len(new_dimensions)):
        if new_dimensions[index] > array.shape[index]:
            # TODO: split the largest nodes
            append_size = tuple(
                new_dimensions[index] - new_values.shape[index] if index == i else new_values.shape[i] for i in
                range(len(new_dimensions)))
            new_values = np.append(new_values,
                                   np.random.normal(scale=var or (1.0 / math.sqrt(float(new_dimensions[index]))),
                                                    loc=mean,
                                                    size=append_size)
                                   .astype(array.dtype),
                                   axis=index)
        elif new_dimensions[index] < new_values.shape[index]:
            # TODO: smarter downsizing
            temp_size = tuple(
                new_dimensions[index] if index == i else new_values.shape[i] for i in
                range(len(new_dimensions)))
            new_values = np.resize(new_values, temp_size)

    assert new_values.shape == new_dimensions
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
                array[split_args] -= np.squeeze(random_noise, axis=[axis])

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
