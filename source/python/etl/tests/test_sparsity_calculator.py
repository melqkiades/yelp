from tripadvisor.fourcity import movielens_extractor

__author__ = 'fpena'

from etl import sparsity_calculator
from tripadvisor.fourcity import extractor
from unittest import TestCase


reviews_matrix_3 = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 1, 'overall_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 3, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 1, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 2, 'overall_rating': 5.0},
    {'user_id': 'U3', 'offering_id': 3, 'overall_rating': 5.0}
]

reviews_matrix_3_1 = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 3, 'overall_rating': 5.0}
]

reviews_matrix_3_2 = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 3, 'overall_rating': 5.0}
]


class TestSparsityCalculator(TestCase):

    def test_get_sparsity(self):

        expected_value = 0.
        actual_value = sparsity_calculator.get_sparsity(reviews_matrix_3)
        self.assertEqual(expected_value, actual_value)

        expected_value = 1 - 6./9
        actual_value = sparsity_calculator.get_sparsity(reviews_matrix_3_1)
        self.assertEqual(expected_value, actual_value)

        expected_value = 1 - 3./9
        actual_value = sparsity_calculator.get_sparsity(reviews_matrix_3_2)
        self.assertEqual(expected_value, actual_value)

    def test_get_sparsity_empty(self):

        self.assertRaises(ValueError, sparsity_calculator.get_sparsity, [])


# movielens_reviews = movielens_extractor.get_ml_100K_dataset()
# fourcity_reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
# print('Movie Lens', sparsity_calculator.get_sparsity(movielens_reviews))
# print('Four City', sparsity_calculator.get_sparsity(fourcity_reviews))