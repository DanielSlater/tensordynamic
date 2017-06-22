import numpy as np

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.data.data_set_collection import DataSetCollection


def xor():
    """
     Returns the two spirals dataset.

    Args:
        noise (float):
        number_of_points (int):
    """
    features = np.array([[1., 0.],
                         [0., 0.],
                         [0., 1.],
                         [1., 1.]])

    labels = np.array([[1.],
                       [0.],
                       [0.],
                       [1.]])

    return features, labels


def get_xor_data_set_collection():
    features, labels = xor()
    train = DataSet(features, labels)
    test = DataSet(features, labels)

    return DataSetCollection("xor", train, test, normalize=False)


if __name__ == '__main__':
    dsc = get_two_spirals_data_set_collection()
    print(dsc.name)