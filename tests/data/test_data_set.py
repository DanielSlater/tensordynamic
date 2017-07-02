import numpy as np
from unittest import TestCase

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.data.data_set_collection import DataSetCollection


class TestDataSet(TestCase):
    def test_num_examples(self):
        data_set = DataSet(np.random.normal(size=(100, 10)), np.random.normal(size=(100, 1)))

        self.assertEqual(data_set.num_examples, 100)

    def test_one_batch_iteration_exact_batch(self):
        batch_size = 10
        data_set = DataSet(np.random.normal(size=(20, 10)), np.random.normal(size=(20, 1)))

        results = list(data_set.one_iteration_in_batches(batch_size))

        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0][0]), batch_size)
        self.assertEqual(len(results[0][1]), batch_size)
        self.assertEqual(len(results[-1][0]), batch_size)
        self.assertEqual(len(results[-1][1]), batch_size)

    def test_one_batch_iteration_exact_partial_batch(self):
        batch_size = 10
        data_set = DataSet(np.random.normal(size=(25, 10)), np.random.normal(size=(25, 1)))

        results = list(data_set.one_iteration_in_batches(batch_size))

        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0][0]), batch_size)
        self.assertEqual(len(results[0][1]), batch_size)
        self.assertEqual(len(results[-1][0]), batch_size)
        self.assertEqual(len(results[-1][1]), batch_size)