import random
from collections import defaultdict

import numpy as np


def parity_fn(input_data):
    ones = 0
    for x in input_data:
        if x > 0.5:
            ones += 1
    return [ones % 2]


def symmetry_fn(input_data):
    for x, y in zip(input_data, reversed(input_data)):
        if x != y:
            return [0.0]
    return [1.0]


def last_bit_fn(input_data):
    if input_data[-1]:
        return [1.0, 0.0]
    else:
        return [0.0, 1.0]


def identity_fn(input_data):
    return input_data


def one_fn(input_data):
    return [1.0, 0.0]


def shuffle(in_x, in_y):
    collection = zip(in_x, in_y)
    random.shuffle(collection)
    out_x = np.array([x for x, y in collection])
    out_y = np.array([y for x, y in collection])
    return out_x, out_y


def _create_dataset(fn, input_size, length, even_classes=False):
    X_train = []
    y_train = []
    class_counts = defaultdict(int)
    for i in range(length):
        X = np.random.binomial(1, 0.5, size=input_size)
        y = fn(X)
        if even_classes:
            y_class = np.argmax(y)
            if i > length / len(y):
                if y_class == max(class_counts, key=class_counts.get):
                    continue
            class_counts[max] += 1

        X_train.append(X)
        y_train.append(y)

    if even_classes:
        collection = zip(X_train, y_train)
        random.shuffle(collection)
        X_train = [x for x, y in collection]
        y_train = [y for x, y in collection]

    return np.array(X_train, dtype='float32') / 1.0, np.array(y_train, dtype='float32') / 1.0


def create_dataset(function, input_size, dataset_size, validation_percent=0.25, test_percent=0.25, even_test_classes=False):
    """
    Create a dataset from a function

    Parameters
    ----------
    function : [float] -> [float]
        function that takes an input of random floats and outputs a transformation of them, can change the size
    input_size : int
        length of array that the function should take as an input
    dataset_size : int
        total number of rows of data to generate across all data sets
    validation_percent : float
        percent of data generated to go into the validation data set
    test_percent : float
        percent of data generated to go into the test data set
    even_test_classes : bool
        If True then it will garentee that all datasets have equal numbers of classes

    Returns
    -------
    numpy.array, numpy.array, numpy.array, numpy.array, numpy.array, numpy.array
    """
    train_x, train_y = _create_dataset(function, input_size, int(dataset_size * (1.0 - validation_percent - test_percent)),
                                       even_classes=even_test_classes)
    val_x, val_y = _create_dataset(function, input_size, int(dataset_size * validation_percent))
    test_x, test_y = _create_dataset(function, input_size, int(dataset_size * test_percent))

    return train_x, train_y, val_x, val_y, test_x, test_y

XOR_INPUTS = np.array([[1.0, -1.0],
                       [-1.0, -1.0],
                       [-1.0, 1.0],
                       [1.0, 1.0]], dtype=np.float32)
XOR_TARGETS = np.array([[1.0],
                        [-1.0],
                        [1.0],
                        [-1.0]], dtype=np.float32)

DOUBLE_XOR_INPUTS = np.array([[-1.0, -1.0, -1.0],
                              [-1.0, -1.0, 1.0],
                              [-1.0, 1.0, -1.0],
                              [-1.0, 1.0, 1.0],
                              [1.0, -1.0, -1.0],
                              [1.0, -1.0, 1.0],
                              [1.0, 1.0, -1.0],
                              [1.0, 1.0, 1.0], ], dtype=np.float32)
DOUBLE_XOR_TARGETS = np.array([[1.0],
                               [1.0],
                               [1.0],
                               [-1.0],
                               [-1.0],
                               [1.0],
                               [1.0],
                               [-1.0]], dtype=np.float32)


def xor_sig_ds():
    return XOR_INPUTS / 2.0 + 0.5, XOR_TARGETS / 2.0 + 0.5


def double_xor_sig_ds():
    return DOUBLE_XOR_INPUTS / 2.0 + 0.5, DOUBLE_XOR_TARGETS / 2.0 + 0.5


def xor_tan_ds():
    return XOR_INPUTS, XOR_TARGETS


def double_xor_tan_ds():
    return DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS


def k_nearest_eculidian_dist(main, others, k=1):
    dists = []

    for item in others:
        diff = main-item
        dist = np.sum(diff*diff)
        dists.append(dist)

    dists.sort()

    return sum(dists[:k])


def pearson_correlation_1vsMany(main, others):
    """data1 & data2 should be numpy arrays."""
    result = []
    main_mean = main.mean()
    main_std = main.std()
    if main_std == 0.0 or main_mean == 0.0:
        return [1000.0] * len(others)

    for data in others:
        mean2 = data.mean()
        std2 = data.std()
        corr = ((main*data).mean()-main_mean*mean2)/(main_std*std2)
        result.append(corr)

    return result


def one_hot(data):
    min_col = min([row[0] for row in data])
    max_col = max([row[0] for row in data])
    range = max_col-min_col
    results = []
    for row in data:
        one_hot_row = [0.0]*(range+1)
        one_hot_row[row[0]-min_col] = 1.0
        results.append(one_hot_row)
    return results


def normalize(data):
    """
    Normalize a dataset so the values in all rows are between 0 and 1

    Parameters
    ----------
    data : [float]

    Returns
    -------
    [float]
    """
    minmax = []
    for i in range(len(data[0])):
        min_col = min([row[i] for row in data])
        max_col = max([row[i] for row in data])
        minmax.append((min_col, max_col, max_col-min_col))

    normalized = []

    for row in data:
        norm_row = [(r-m[0])/m[2] for r, m in zip(row, minmax)]
        normalized.append(norm_row)

    return normalized

if __name__ == '__main__':
    train_x, train_y, _, _, _, _ = create_dataset(symmetry_fn, 10, 100, even_test_classes=True)
    print np.mean(train_y)
