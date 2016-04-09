import math

import numpy as np
import tensorflow as tf


def noise_weight_extender(old_values, new_dims):
    '''

    Parameters
    ----------
    old_values
    new_dims

    Returns
    -------

    '''
    new_values = old_values

    # bias
    if len(new_dims) == 1:
        if new_dims[0] > old_values.shape[-1]:
            new_values = np.append(old_values, np.zeros([new_dims[0] - old_values.shape[-1]])) \
                .astype(old_values.dtype)
    else:
        if new_dims[0] > old_values.shape[0]:
            new_values = np.append(new_values,
                                   np.random.normal(scale=1.0 / math.sqrt(float(new_dims[0])),
                                                    size=(new_dims[0] - new_values.shape[0], new_values.shape[1]))
                                   .astype(old_values.dtype),
                                   axis=0)
        if new_dims[1] > old_values.shape[1]:
            new_values = np.append(new_values,
                                   np.random.normal(scale=(1.0 / math.sqrt(float(new_dims[1])) / 100.),
                                                    size=(new_values.shape[0], new_dims[1] - new_values.shape[1]))
                                   .astype(old_values.dtype),
                                   axis=1)

    return new_values
