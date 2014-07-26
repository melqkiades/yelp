from unittest import TestCase
from evaluation.root_mean_square_error import RootMeanSquareError

__author__ = 'fpena'


class TestRootMeanSquareError(TestCase):

    def test_compute_list(self):
        errors = [0.7, 0.1, 0.1, 1.1, 1.5]
        self.assertEqual(RootMeanSquareError.compute_list(errors), 0.8910667763978186)
        errors = [0, 0]
        self.assertEqual(RootMeanSquareError.compute_list(errors), 0)
        errors = [0.7, 0.1, 0.1, None, 1.5]
        self.assertEqual(RootMeanSquareError.compute_list(errors), 0.8306623862918074)
        errors = []
        self.assertEqual(RootMeanSquareError.compute_list(errors), None)
