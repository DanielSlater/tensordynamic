import numpy as np

from unittest import TestCase

from tensor_dynamic.weight_functions import array_extend, noise_weight_extender


class TestWeightFunctions(TestCase):
    def test_array_split_extention_axis_1(self):
        a = np.array([[1, 2, 3],
                      [4, 5, 6]])

        split_extended = array_extend(a, {1: [1]})

        np.testing.assert_array_almost_equal(split_extended, np.array([[1, 2, 3, 2], [4, 5, 6, 5]]))

    def test_array_split_extention_axis_2(self):
        a = np.array([[1, 2, 3],
                      [4, 5, 6]])

        split_extended = array_extend(a, {0: [0]})

        np.testing.assert_array_almost_equal(split_extended, np.array([[1, 2, 3],
                                                                       [4, 5, 6],
                                                                       [1, 2, 3]]))

    def test_array_split_extention_vector(self):
        a = np.array([1, 2, 3])

        split_extended = array_extend(a, {0: [0]})

        np.testing.assert_array_almost_equal(split_extended, np.array([1, 2, 3, 1]))

    def test_array_split_extention_halve_splits(self):
        a = np.array([[2., 4., 8.],
                      [1., 2., 3.]])

        split_extended = array_extend(a, {0: [0]}, halve_extended_vectors=True)

        np.testing.assert_array_almost_equal(split_extended, np.array([[1., 2., 4.],
                                                                       [1., 2., 3.],
                                                                       [1., 2., 4.]]))

    def test_noise_weight_extender_shrink(self):
        a = np.array([[2., 4., 8.],
                      [1., 2., 3.]])

        b = noise_weight_extender(a, (2, 2))

        self.assertEqual(b.shape, (2, 2))