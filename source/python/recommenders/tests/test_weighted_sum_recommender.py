from unittest import TestCase
from evaluation.mean_absolute_error import MeanAbsoluteError
from evaluation.root_mean_square_error import RootMeanSquareError
from recommenders.weighted_sum_recommender import WeightedSumRecommender
from tripadvisor.fourcity import four_city_evaluator
from tripadvisor.fourcity.dummy_predictor import DummyPredictor

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


class TestWeightedSumRecommender(TestCase):

    def test_predict_rating(self):

        recommender = WeightedSumRecommender()
        recommender.load(reviews)

        actual_rating_1 = 4.0
        self.assertEqual(actual_rating_1, recommender.predict_rating('A1', 1))
        actual_rating_2 = 2.0
        self.assertEqual(actual_rating_2, recommender.predict_rating('A1', 3))

    def test_compare_against_dummy_recommender(self):
        recommender = WeightedSumRecommender()
        recommender.load(reviews_matrix_5)
        _, errors = four_city_evaluator.predict_rating_list(recommender, reviews_matrix_5)
        wsr_mean_absolute_error = MeanAbsoluteError.compute_list(errors)
        wsr_root_mean_square_error = RootMeanSquareError.compute_list(errors)
        print('Mean Absolute error:', wsr_mean_absolute_error)
        print('Root mean square error:',  wsr_root_mean_square_error)

        recommender = DummyPredictor(6.0)
        _, errors = four_city_evaluator.predict_rating_list(recommender, reviews_matrix_5)
        dummy_mean_absolute_error = MeanAbsoluteError.compute_list(errors)
        dummy_root_mean_square_error = RootMeanSquareError.compute_list(errors)
        print('Mean Absolute error:', dummy_mean_absolute_error)
        print('Root mean square error:',  dummy_root_mean_square_error)

        self.assertLess(wsr_mean_absolute_error, dummy_mean_absolute_error)
