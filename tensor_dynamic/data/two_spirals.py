import numpy as np

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.data.data_set_collection import DataSetCollection


def two_spirals(number_of_points, noise=.5):
    """
     Returns the two spirals dataset.

    Args:
        noise (float):
        number_of_points (int):
    """
    points_per_class = number_of_points / 2
    n = np.sqrt(np.random.rand(points_per_class, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(points_per_class, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(points_per_class, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(points_per_class), np.ones(points_per_class))).reshape(number_of_points, 1))


def get_two_spirals_data_set_collection():
    train_features, train_labels = two_spirals(2000)
    test_features, test_labels = two_spirals(1000)
    train = DataSet(train_features, train_labels)
    test = DataSet(test_features, test_labels)

    return DataSetCollection("two spirals", train, test, normalize=False)


if __name__ == '__main__':
    dsc = get_two_spirals_data_set_collection()
    print(dsc.name)