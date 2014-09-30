from unittest import TestCase
from tripadvisor.reviews_dataset_analyzer import ReviewsDatasetAnalyzer

__author__ = 'fpena'


reviews = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 4.0},
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 4.0},
    {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 1, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 3, 'overall_rating': 4.0},
    {'user_id': 'U3', 'offering_id': 2, 'overall_rating': 4.0},
    {'user_id': 'U4', 'offering_id': 2, 'overall_rating': 4.0},
    {'user_id': 'U4', 'offering_id': 3, 'overall_rating': 4.0},
    {'user_id': 'U5', 'offering_id': 2, 'overall_rating': 4.0},
    {'user_id': 'U6', 'offering_id': 2, 'overall_rating': 4.0},
    {'user_id': 'U7', 'offering_id': 2, 'overall_rating': 4.0},
    {'user_id': 'U8', 'offering_id': 1, 'overall_rating': 4.0},
    {'user_id': 'U8', 'offering_id': 2, 'overall_rating': 4.0}
]

reviews_small = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 4.0}
]

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


class TestReviewsDatasetAnalyzer(TestCase):

    def test_count_items_in_common(self):

        expected_value = {1: 23, 2: 4, 3: 1}
        rda = ReviewsDatasetAnalyzer(reviews)
        actual_value = rda.count_items_in_common()
        self.assertEqual(expected_value, actual_value)

        expected_value = {0: 1}
        rda = ReviewsDatasetAnalyzer(reviews_small)
        actual_value = rda.count_items_in_common()
        self.assertEqual(expected_value, actual_value)

    def test_calculate_sparsity(self):

        expected_value = 0.
        rda = ReviewsDatasetAnalyzer(reviews_matrix_3)
        actual_value = rda.calculate_sparsity()
        self.assertEqual(expected_value, actual_value)

        expected_value = 1 - 6./9
        rda = ReviewsDatasetAnalyzer(reviews_matrix_3_1)
        actual_value = rda.calculate_sparsity()
        self.assertEqual(expected_value, actual_value)

        expected_value = 1 - 3./9
        rda = ReviewsDatasetAnalyzer(reviews_matrix_3_2)
        actual_value = rda.calculate_sparsity()
        self.assertEqual(expected_value, actual_value)

    def test_init_empty(self):

        self.assertRaises(ValueError, ReviewsDatasetAnalyzer, [])



