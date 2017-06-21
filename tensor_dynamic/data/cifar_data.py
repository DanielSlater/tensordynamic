"""Functions for downloading and reading MNIST data."""
from __future__ import print_function

import cPickle as pickle
import numpy as np
import os

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.data.data_set_collection import DataSetCollection
from tensor_dynamic.data.mnist_data import dense_to_one_hot


def _get_CIFAR10_data(cifar10_dir, num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the neural net classifier.
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = _load(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2)
    X_val = X_val.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)

    mean_image = np.mean(X_train, axis=0)
    std = np.std(X_train)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train /= std
    X_val /= std
    X_test /= std

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'mean': mean_image, 'std': std
    }


def _load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'r') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def _load(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = _load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = _load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_cifar_10_data_set_collection(root_path, one_hot=True):
    """Get the cifar 100 data set requires files to be downloaded and extracted into cifar-10-batches-py
    directory within root path

    Args:
        root_path (str):
        one_hot (bool): If True converts sparse labels to one hot encoding

    Returns:
        DataSetCollection
    """
    root_path += "/cifar-10-batches-py"

    features_train, labels_train, features_test, labels_test = _load(root_path)

    if one_hot:
        labels_train = dense_to_one_hot(labels_train)
        labels_test = dense_to_one_hot(labels_test)

    train = DataSet(features_train, labels_train, to_binary=True)

    test = DataSet(features_test, labels_test, to_binary=True)

    collection = DataSetCollection(train, test, normalize=True)

    return collection


def get_cifar_100_data_set_collection(root_path, one_hot=True, use_fine_labels=True):
    """Get the cifar 100 data set requires files to be downloaded and extracted into cifar-100-python
    directory within root path

    Args:
        root_path (str):
        one_hot (bool): If True converts sparse labels to one hot encoding
        use_fine_labels (bool): If true use full 100 labels, if False use 10 categories

    Returns:
        DataSetCollection
    """
    root_path = root_path + "/cifar-100-python"

    features_train, labels_train = _load_cifar_100_set(root_path + "/train", use_fine_labels)
    features_test, labels_test = _load_cifar_100_set(root_path + "/test", use_fine_labels)

    if one_hot:
        num_classes = 100 if use_fine_labels else 10
        labels_train = dense_to_one_hot(labels_train, num_classes)
        labels_test = dense_to_one_hot(labels_test, num_classes)

    train = DataSet(features_train, labels_train, to_binary=True)

    test = DataSet(features_test, labels_test, to_binary=True)

    collection = DataSetCollection(train, test, normalize=True)

    return collection


def _load_cifar_100_set(filepath, use_fine_labels):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    features = data['data'].reshape(data['data'].shape[0],
                                    3, 32, 32)
    if use_fine_labels:
        labels = np.array(data['fine_labels'],
                          dtype=np.uint8)
    else:
        labels = np.array(data['coarse_labels'],
                          dtype=np.uint8)

    return features, labels


if __name__ == '__main__':
    data_set = get_cifar_100_data_set_collection("CIFAR_data", one_hot=True)
    print(len(data_set))
    data_set = get_cifar_10_data_set_collection("CIFAR_data", one_hot=True)
    print(len(data_set))
