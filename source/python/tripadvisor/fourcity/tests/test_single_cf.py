from unittest import TestCase

from evaluation.mean_absolute_error import MeanAbsoluteError
from evaluation.root_mean_square_error import RootMeanSquareError
from tripadvisor.fourcity import four_city_evaluator
from recommenders.dummy_recommender import DummyRecommender
from tripadvisor.fourcity.single_cf import SingleCF


__author__ = 'fpena'


reviews = [
    {
        'user_id': 'A1',
        'offering_id': 1,
        'overall_rating': 4.0
    },
    {
        'user_id': 'A1',
        'offering_id': 1,
        'overall_rating': 5.0
    },
    {
        'user_id': 'A1',
        'offering_id': 2,
        'overall_rating': 2.0
    },
    {
        'user_id': 'A1',
        'offering_id': 3,
        'overall_rating': 3.0
    },
    {
        'user_id': 'A2',
        'offering_id': 1,
        'overall_rating': 4.0
    },
    {
        'user_id': 'A2',
        'offering_id': 3,
        'overall_rating': 2.0
    }

]


reviews_matrix_5 = [
    # {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 4, 'overall_rating': 7.0},
    # {'user_id': 'U1', 'offering_id': 5, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 4, 'overall_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 5, 'overall_rating': 9.0},
    {'user_id': 'U3', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U3', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U3', 'offering_id': 4, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 5, 'overall_rating': 9.0},
    {'user_id': 'U4', 'offering_id': 1, 'overall_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 2, 'overall_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 3, 'overall_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 4, 'overall_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 5, 'overall_rating': 5.0},
    {'user_id': 'U5', 'offering_id': 1, 'overall_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 2, 'overall_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 3, 'overall_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 4, 'overall_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 5, 'overall_rating': 5.0},
]


class TestSingleCF(TestCase):

    def test_calculate_adjusted_weighted_sum(self):

        single_cf = SingleCF()
        single_cf.load(reviews)

        actual_rating_1 = 3.5 + 0.99227787671366752 * (4.0 - 3.0) / 0.99227787671366752
        self.assertEqual(actual_rating_1, single_cf.calculate_adjusted_weighted_sum('A1', 1))
        actual_rating_2 = 3.5 + 0.99227787671366752 * (2.0 - 3.0) / 0.99227787671366752
        self.assertEqual(actual_rating_2, single_cf.calculate_adjusted_weighted_sum('A1', 3))

    def test_calculate_weighted_sum(self):

        single_cf = SingleCF()
        single_cf.load(reviews)

        actual_rating_1 = 4.0
        self.assertEqual(actual_rating_1, single_cf.calculate_weighted_sum('A1', 1))
        actual_rating_2 = 2.0
        self.assertEqual(actual_rating_2, single_cf.calculate_weighted_sum('A1', 3))

    def test_compare_against_dummy_recommender(self):
        clusterer = SingleCF()
        clusterer.load(reviews_matrix_5)
        _, errors = four_city_evaluator.predict_rating_list(clusterer, reviews_matrix_5)
        single_mean_absolute_error = MeanAbsoluteError.compute_list(errors)
        single_root_mean_square_error = RootMeanSquareError.compute_list(errors)
        print('Mean Absolute error:', single_mean_absolute_error)
        print('Root mean square error:',  single_root_mean_square_error)

        clusterer = DummyRecommender(6.0)
        _, errors = four_city_evaluator.predict_rating_list(clusterer, reviews_matrix_5)
        dummy_mean_absolute_error = MeanAbsoluteError.compute_list(errors)
        dummy_root_mean_square_error = RootMeanSquareError.compute_list(errors)
        print('Mean Absolute error:', dummy_mean_absolute_error)
        print('Root mean square error:',  dummy_root_mean_square_error)

        self.assertLess(single_mean_absolute_error, dummy_mean_absolute_error)
