"""Functions for downloading and reading MNIST data."""
from __future__ import print_function

import cPickle as pickle
import numpy as np
import os

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.data.data_set_collection import DataSetCollection
from tensor_dynamic.data.mnist_data import dense_to_one_hot

CIFAR_DATA_DIR = os.path.dirname(__file__) + "/CIFAR_data"


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


def get_cifar_10_data_set_collection(root_path=CIFAR_DATA_DIR, one_hot=True,
                                     validation_size=0,
                                     validation_ratio=None):
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

    if not validation_size and validation_ratio:
        validation_size = (len(labels_train) + len(labels_test)) * validation_ratio

    if validation_size:
        features_validation = features_train[validation_size:]
        labels_validation = labels_train[validation_size:]

        features_train = features_train[validation_size:]
        labels_train = labels_train[validation_size:]
        validation = DataSet(features_validation, labels_validation, to_binary=True)
    else:
        validation = None

    train = DataSet(features_train, labels_train, to_binary=True)

    test = DataSet(features_test, labels_test, to_binary=True)

    collection = DataSetCollection('CIFAR-10', train, test, validation=validation, normalize=True)

    return collection


def get_cifar_100_data_set_collection(root_path=CIFAR_DATA_DIR, one_hot=True, use_fine_labels=True,
                                      validation_size=0,
                                      validation_ratio=None):
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

    if not validation_size and validation_ratio:
        validation_size = (len(labels_train) + len(labels_test)) * validation_ratio

    if validation_size:
        features_validation = features_train[validation_size:]
        labels_validation = labels_train[validation_size:]

        features_train = features_train[validation_size:]
        labels_train = labels_train[validation_size:]
        validation = DataSet(features_validation, labels_validation, to_binary=True)
    else:
        validation = None

    train = DataSet(features_train, labels_train, to_binary=True)

    test = DataSet(features_test, labels_test, to_binary=True)

    collection = DataSetCollection('CIFAR-100' + ('-fine' if use_fine_labels else '-coarse'),
                                   train, test, validation=validation, normalize=True)

    return collection


def _load_cifar_100_set(filepath, use_fine_labels):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    features = data['data'].reshape(data['data'].shape[0],
                                    3, 32, 32)

    # change from channel, width, height to width, height, channel
    features = features.transpose(0, 2, 3, 1)

    features = features.astype(np.float32)

    if use_fine_labels:
        labels = np.array(data['fine_labels'],
                          dtype=np.uint8)
    else:
        labels = np.array(data['coarse_labels'],
                          dtype=np.uint8)

    return features, labels


# TODO: Fix and maybe use this in the future
def _maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    import sys
    import urllib
    import tarfile

    DATA_URL_10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    DATA_URL_100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'

    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL_10.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL_10, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


if __name__ == '__main__':
    # data_set = _get_CIFAR10_data("CIFAR_data/cifar-10-batches-py")
    data_set = get_cifar_100_data_set_collection(CIFAR_DATA_DIR, one_hot=True, validation_ratio=.2)
    data_set = get_cifar_10_data_set_collection(CIFAR_DATA_DIR, one_hot=True, validation_ratio=.2)
    print(data_set.name)
