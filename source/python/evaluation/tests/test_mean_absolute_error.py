from unittest import TestCase
from evaluation.mean_absolute_error import MeanAbsoluteError

__author__ = 'fpena'


class TestMeanAbsoluteError(TestCase):

    def test_compute_list(self):
        errors = [0.7, 0.1, 0.1, 1.1, 1.5]
        self.assertEqual(MeanAbsoluteError.compute_list(errors), 0.7)
        errors = [0, 0]
        self.assertEqual(MeanAbsoluteError.compute_list(errors), 0)
        errors = [0.7, 0.1, 0.1, None, 1.5]
        self.assertEqual(MeanAbsoluteError.compute_list(errors), 0.6)
        errors = []
        self.assertEqual(MeanAbsoluteError.compute_list(errors), None)
