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

    def test_array_split_extention_axis_3(self):
        a = np.array([[[1, 2], [3, 4]],
                      [[5, 6], [7, 8]]])

        split_extended = array_extend(a, {2: [0]})

        np.testing.assert_array_almost_equal(split_extended, np.array([[[1, 2, 1], [3, 4, 3]],
                                                                       [[5, 6, 5], [7, 8, 7]]]))

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

    def test_noise_weight_extender_4_dim(self):
        a = np.random.normal(size=(5, 4, 3, 2))

        new_dimensions = (5, 4, 3, 3)
        b = noise_weight_extender(a, new_dimensions)

        self.assertEqual(b.shape, new_dimensions)

    def test_noise_weight_extender_4_dim_2(self):
        a = np.random.normal(size=(5, 4, 3, 3))

        new_dimensions = (5, 4, 3, 1)
        b = noise_weight_extender(a, new_dimensions)

        self.assertEqual(b.shape, new_dimensions)

    def test_noise_weight_extender_4_dim_3(self):
        a = np.random.normal(size=(5, 4, 3, 2))

        new_dimensions = (2, 3, 4, 5)
        b = noise_weight_extender(a, new_dimensions)

        self.assertEqual(b.shape, new_dimensions)

    def test_noise_weight_extender_1_dim(self):
        a = np.random.normal(size=(5,))

        new_dimensions = (10,)
        b = noise_weight_extender(a, new_dimensions)

        self.assertEqual(b.shape, new_dimensions)