from unittest import TestCase
from recommenders.multicriteria.overall_cf_recommender import \
    OverallCFRecommender

__author__ = 'fpena'


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


class TestOverallCFRecommender(TestCase):

    def test_predict_rating(self):
        recommender = OverallCFRecommender(significant_criteria_ranges=None)
        recommender.load(reviews_matrix_5)

        # print(recommender.user_similarity_matrix['U1']['U2'])
        # print(recommender.user_similarity_matrix['U1']['U3'])
        # print(recommender.user_similarity_matrix['U1']['U4'])
        # print(recommender.user_similarity_matrix['U1']['U5'])

        numerator = (0.291666666667 * 9) * 2 + (0.142857142857 * 5) * 2
        denominator = (0.291666666667 * 2) + (0.142857142857 * 2)
        expected_result = numerator / denominator

        self.assertAlmostEqual(expected_result, recommender.predict_rating('U1', 5), 7)
        self.assertEqual(6.2173913043478271, recommender.predict_rating('U4', 5))
        self.assertEqual(None, recommender.predict_rating('U4', 6))

        recommender.load(reviews_matrix_5[3:])
        self.assertEqual(7.9768889264474412, recommender.predict_rating('U1', 5))
