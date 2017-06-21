from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.data.semi_data_set import SemiDataSet


class DataSetCollection(object):
    def __init__(self, train, test, validation=None):
        """Collects data for doing full training and validation of a model

        Args:
            train (DataSet): features and labels used for training
            test  (DataSet): features and labels used for testing
            validation (DataSet): optional features and labels used for validation
        """
        assert isinstance(train, (DataSet, SemiDataSet))
        assert isinstance(test, (DataSet, SemiDataSet))
        if validation is not None:
            assert isinstance(validation, (DataSet, SemiDataSet))
        self._train = train
        self._test = test
        self._validation = validation

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def validation(self):
        return self._validation