from unittest import TestCase

from evaluation.mean_absolute_error import MeanAbsoluteError
from evaluation.root_mean_square_error import RootMeanSquareError
from recommenders.adjusted_weighted_sum_recommender import \
    AdjustedWeightedSumRecommender
from recommenders.similarity.single_similarity_matrix_builder import \
    SingleSimilarityMatrixBuilder
from tripadvisor.fourcity import four_city_evaluator
from recommenders.dummy_recommender import DummyRecommender


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


class TestAdjustedWeightedSumRecommender(TestCase):

    def test_predict_rating(self):

        recommender = AdjustedWeightedSumRecommender(SingleSimilarityMatrixBuilder('cosine'))
        recommender.load(reviews)

        actual_rating_1 = 3.5 + 0.99227787671366752 * (4.0 - 3.0) / 0.99227787671366752
        self.assertEqual(actual_rating_1, recommender.predict_rating('A1', 1))
        actual_rating_2 = 3.5 + 0.99227787671366752 * (2.0 - 3.0) / 0.99227787671366752
        self.assertEqual(actual_rating_2, recommender.predict_rating('A1', 3))
