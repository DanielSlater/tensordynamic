import numpy as np

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.data.semi_data_set import SemiDataSet


class DataSetCollection(object):
    def __init__(self, name, train, test, validation=None, normalize=True):
        """Collects data for doing full training and validation of a model

        Args:
            name (str): name for this data set, used for reporting results
            normalize (bool): If True data is normalized, by taking the mean and std of the training set
                and applying it to all other data sets
            train (DataSet): features and labels used for training
            test  (DataSet): features and labels used for testing
            validation (DataSet): optional features and labels used for validation
        """
        assert isinstance(name, str)
        assert isinstance(train, (DataSet, SemiDataSet))
        assert isinstance(test, (DataSet, SemiDataSet))
        assert train.features.shape[1:] == test.features.shape[1:]
        assert train.labels.shape[1:] == test.labels.shape[1:]
        if validation is not None:
            assert isinstance(validation, (DataSet, SemiDataSet))
            assert train.features.shape[1:] == validation.features.shape[1:]
            assert train.labels.shape[1:] == validation.labels.shape[1:]

        if normalize:
            mean_image = np.mean(train.features, axis=0)
            std = np.std(train.features, axis=0)
            std += 1e-10

            train._features -= mean_image
            test._features -= mean_image
            train._features /= std
            test._features /= std

            if validation:
                validation._features -= mean_image
                validation._features /= std

        self._train = train
        self._test = test
        self._validation = validation
        self._name = name
        self._normalized = normalize

    @property
    def normlized(self):
        return self._normalized

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def validation(self):
        return self._validation

    @property
    def name(self):
        return self._name

    @property
    def features_shape(self):
        """Shape of a single instance of features for the dataset, ignores batch dimension

        Returns:
            (int)
        """
        return self._train.features.shape[1:]

    @property
    def labels_shape(self):
        """Shape of a single instance of labels for the dataset, ignores batch dimension

        Returns:
            (int)
        """
        return self._train.labels.shape[1:]