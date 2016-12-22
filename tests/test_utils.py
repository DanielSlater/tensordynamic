from unittest import TestCase

from tensor_dynamic.utils import train_till_convergence


class TestUtils(TestCase):
    def test_train_till_convergence(self):
        FINAL_ERROR = 3
        errors = [5, 4, 3, 2, 2, 1, 2, 2, FINAL_ERROR]
        errors_iter = iter(errors)

        final_error = train_till_convergence(lambda: next(errors_iter), continue_epochs=3)

        self.assertEqual(final_error, FINAL_ERROR)