from unittest import TestCase
from evaluation import precision_in_top_n
from recommenders.item_average_recommender import ItemAverageRecommender
from recommenders.similarity.single_similarity_matrix_builder import \
    SingleSimilarityMatrixBuilder
from recommenders.weighted_sum_recommender import WeightedSumRecommender

__author__ = 'fpena'


known_ratings_1 = {
    'I1': 5.0,
    'I2': 3.0,
    'I3': 5.0,
    'I4': 4.0,
    'I5': 1.0,
    'I6': 2.0,
    'I7': 3.5,
    'I8': 3.0,
    'I9': 5.0,
    'I10': 2.0
}

predicted_ratings_1 = {
    'I1': 3.0,
    'I2': 5.0,
    'I3': 4.4,
    'I4': 4.2,
    'I5': 3.2,
    'I6': 2.0,
    'I7': 4.3,
    'I8': 3.1,
    'I9': 4.1,
    'I10': 1.0
}

known_ratings_2 = {
    'I1': 5.0,
    'I2': 3.0,
    'I3': 5.0,
}

predicted_ratings_2 = {
    'I1': 3.0,
    'I2': 5.0,
    'I3': 4.4,
}

known_ratings_3 = {
    'I1': 2.0,
    'I2': 3.0,
    'I3': 1.0,
}

predicted_ratings_3 = {
    'I1': 3.0,
    'I2': 5.0,
    'I3': 4.4,
}

reviews_matrix_5 = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0, 'multi_ratings': [2.0, 2.0, 8.0, 8.0, 5.0]},
    {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 7.0, 'multi_ratings': [5.0, 5.0, 9.0, 9.0, 7.0]},
    {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 5.0, 'multi_ratings': [2.0, 2.0, 8.0, 8.0, 5.0]},
    {'user_id': 'U1', 'offering_id': 4, 'overall_rating': 7.0, 'multi_ratings': [5.0, 5.0, 9.0, 9.0, 7.0]},
    # {'user_id': 'U1', 'offering_id': 5, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 1, 'overall_rating': 5.0, 'multi_ratings': [8.0, 8.0, 2.0, 2.0, 5.0]},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0, 'multi_ratings': [9.0, 9.0, 5.0, 5.0, 7.0]},
    {'user_id': 'U2', 'offering_id': 3, 'overall_rating': 5.0, 'multi_ratings': [8.0, 8.0, 2.0, 2.0, 5.0]},
    {'user_id': 'U2', 'offering_id': 4, 'overall_rating': 7.0, 'multi_ratings': [9.0, 9.0, 5.0, 5.0, 7.0]},
    {'user_id': 'U2', 'offering_id': 5, 'overall_rating': 9.0, 'multi_ratings': [9.0, 9.0, 9.0, 9.0, 9.0]},
    {'user_id': 'U3', 'offering_id': 1, 'overall_rating': 5.0, 'multi_ratings': [8.0, 8.0, 2.0, 2.0, 5.0]},
    {'user_id': 'U3', 'offering_id': 2, 'overall_rating': 7.0, 'multi_ratings': [9.0, 9.0, 5.0, 5.0, 7.0]},
    {'user_id': 'U3', 'offering_id': 3, 'overall_rating': 5.0, 'multi_ratings': [8.0, 8.0, 2.0, 2.0, 5.0]},
    {'user_id': 'U3', 'offering_id': 4, 'overall_rating': 7.0, 'multi_ratings': [9.0, 9.0, 5.0, 5.0, 7.0]},
    {'user_id': 'U3', 'offering_id': 5, 'overall_rating': 9.0, 'multi_ratings': [9.0, 9.0, 9.0, 9.0, 9.0]},
    {'user_id': 'U4', 'offering_id': 1, 'overall_rating': 6.0, 'multi_ratings': [3.0, 3.0, 9.0, 9.0, 6.0]},
    {'user_id': 'U4', 'offering_id': 2, 'overall_rating': 6.0, 'multi_ratings': [3.0, 3.0, 9.0, 9.0, 6.0]},
    {'user_id': 'U4', 'offering_id': 3, 'overall_rating': 6.0, 'multi_ratings': [4.0, 4.0, 8.0, 8.0, 6.0]},
    {'user_id': 'U4', 'offering_id': 4, 'overall_rating': 6.0, 'multi_ratings': [4.0, 4.0, 8.0, 8.0, 6.0]},
    {'user_id': 'U4', 'offering_id': 5, 'overall_rating': 5.0, 'multi_ratings': [5.0, 5.0, 5.0, 5.0, 5.0]},
    {'user_id': 'U5', 'offering_id': 1, 'overall_rating': 6.0, 'multi_ratings': [3.0, 3.0, 9.0, 9.0, 6.0]},
    {'user_id': 'U5', 'offering_id': 2, 'overall_rating': 6.0, 'multi_ratings': [3.0, 3.0, 9.0, 9.0, 6.0]},
    {'user_id': 'U5', 'offering_id': 3, 'overall_rating': 6.0, 'multi_ratings': [4.0, 4.0, 8.0, 8.0, 6.0]},
    {'user_id': 'U5', 'offering_id': 4, 'overall_rating': 6.0, 'multi_ratings': [4.0, 4.0, 8.0, 8.0, 6.0]},
    {'user_id': 'U5', 'offering_id': 5, 'overall_rating': 5.0, 'multi_ratings': [5.0, 5.0, 5.0, 5.0, 5.0]}
]


class TestPrecisionInTopN(TestCase):

    def test_calculate_precision(self):

        expected_precision = 0.
        actual_precision = precision_in_top_n.calculate_precision(
            known_ratings_1, predicted_ratings_1, 1, 4.0)
        self.assertEqual(expected_precision, actual_precision)

        expected_precision = 1./3
        actual_precision = precision_in_top_n.calculate_precision(
            known_ratings_1, predicted_ratings_1, 3, 4.0)
        self.assertEqual(expected_precision, actual_precision)

        expected_precision = 3./5
        actual_precision = precision_in_top_n.calculate_precision(
            known_ratings_1, predicted_ratings_1, 5, 4.0)
        self.assertEqual(expected_precision, actual_precision)

        expected_precision = 3./7
        actual_precision = precision_in_top_n.calculate_precision(
            known_ratings_1, predicted_ratings_1, 7, 4.0)
        self.assertEqual(expected_precision, actual_precision)

        expected_precision = 0.
        actual_precision = precision_in_top_n.calculate_precision(
            known_ratings_2, predicted_ratings_2, 1, 4.0)
        self.assertEqual(expected_precision, actual_precision)

        expected_precision = 2./3
        actual_precision = precision_in_top_n.calculate_precision(
            known_ratings_2, predicted_ratings_2, 3, 4.0)
        self.assertEqual(expected_precision, actual_precision)

        expected_precision = 2./3
        actual_precision = precision_in_top_n.calculate_precision(
            known_ratings_2, predicted_ratings_2, 5, 4.0)
        self.assertEqual(expected_precision, actual_precision)

    def test_calculate_precision_none(self):

        expected_precision = None
        actual_precision = precision_in_top_n.calculate_precision([], [], 5, 4.0)
        self.assertEqual(expected_precision, actual_precision)

    def test_calculate_recommender_precision(self):

        recommender = WeightedSumRecommender(SingleSimilarityMatrixBuilder('euclidean'))
        recommender.load(reviews_matrix_5)
        # print(precision_in_top_n.calculate_recommender_precision(
        #     reviews_matrix_5, 'U1', recommender, 3, 8.0))
        # print(precision_in_top_n.calculate_recommender_precision(
        #     reviews_matrix_5, 'U2', recommender, 5, 8.0))
        # print(precision_in_top_n.calculate_recommender_precision(
        #     reviews_matrix_5, 'U3', recommender, 7, 8.0))

    def test_calculate_recall_in_top_n(self):
        recommender = ItemAverageRecommender()
        recommender.load(reviews_matrix_5)

        actual_value = precision_in_top_n.calculate_recall_in_top_n(
            reviews_matrix_5, recommender, 2, 2, None, 4.0)['Top N']
        expected_value = 0.875
        self.assertEqual(expected_value, actual_value)

    def test_is_a_hit(self):

        actual_value = precision_in_top_n.is_a_hit('I2', predicted_ratings_2, 1)
        expected_value = True
        self.assertEqual(expected_value, actual_value)

        actual_value = precision_in_top_n.is_a_hit('I2', predicted_ratings_2, 3)
        expected_value = True
        self.assertEqual(expected_value, actual_value)

        actual_value = precision_in_top_n.is_a_hit('I2', predicted_ratings_2, 10)
        expected_value = True
        self.assertEqual(expected_value, actual_value)

        actual_value = precision_in_top_n.is_a_hit('I1', predicted_ratings_2, 1)
        expected_value = False
        self.assertEqual(expected_value, actual_value)

        actual_value = precision_in_top_n.is_a_hit('I1', predicted_ratings_2, 3)
        expected_value = True
        self.assertEqual(expected_value, actual_value)

        actual_value = precision_in_top_n.is_a_hit('I3', predicted_ratings_2, 2)
        expected_value = True
        self.assertEqual(expected_value, actual_value)

        actual_value = precision_in_top_n.is_a_hit('I3', predicted_ratings_2, 2)
        expected_value = True
        self.assertEqual(expected_value, actual_value)

