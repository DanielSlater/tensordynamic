"""Functions for downloading and reading MNIST data."""
from __future__ import print_function

import gzip
import os
import urllib

import numpy

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.data.data_set_collection import DataSetCollection
from tensor_dynamic.data.semi_data_set import SemiDataSet

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def _maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)


def _extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def _extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


def get_mnist_data_set_collection(train_dir=os.path.dirname(__file__) + "/MNIST_data",
                                  number_labeled_examples=None,
                                  one_hot=True,
                                  validation_size=0,
                                  validation_ratio=None,
                                  limit_train_size=None,
                                  flatten=True):
    """Load mnist data

    Args:
        train_dir (str): directory to store the downloaded data, or to where it has previously been downloaded
        number_labeled_examples (int): For semi supervised learning, how many labels to use, if None we use supervised
            learning
        one_hot (bool): If True labels will be one hot vectors, not ints
        validation_size (int): Number of items to move to validation set
        limit_train_size (int): If set limit number of training items to this
        flatten (bool): If true data set is flattened to simply be array of image values, not 3d array of
            [width, height, depth]

    Returns:
        DataSetCollection
    """
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = _maybe_download(TRAIN_IMAGES, train_dir)
    train_images = _extract_images(local_file)

    local_file = _maybe_download(TRAIN_LABELS, train_dir)
    train_labels = _extract_labels(local_file, one_hot=one_hot)

    local_file = _maybe_download(TEST_IMAGES, train_dir)
    test_images = _extract_images(local_file)

    local_file = _maybe_download(TEST_LABELS, train_dir)
    test_labels = _extract_labels(local_file, one_hot=one_hot)

    if not validation_size and validation_ratio:
        validation_size = (len(train_labels) + len(test_labels)) * validation_ratio

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]

    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    if limit_train_size:
        train_images = train_images[:limit_train_size]
        train_labels = train_labels[:limit_train_size]

    if number_labeled_examples is None:
        train = DataSet(train_images, train_labels, flatten=flatten, to_binary=True)
    else:
        train = SemiDataSet(train_images, train_labels, number_labeled_examples)

    test = DataSet(test_images, test_labels, flatten=flatten, to_binary=True)

    if validation_size:
        validation = DataSet(validation_images, validation_labels, flatten=flatten, to_binary=True)
    else:
        validation = None

    return DataSetCollection('MNIST', train, test, validation)


if __name__ == '__main__':
    mnist = get_mnist_data_set_collection("MNIST_data", one_hot=True, validation_ratio=.2)
    print(mnist.name)
