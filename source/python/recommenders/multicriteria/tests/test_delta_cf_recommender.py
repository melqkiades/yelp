from unittest import TestCase
from recommenders.multicriteria.delta_cf_recommender import DeltaCFRecommender

__author__ = 'fpena'


reviews_matrix_5 = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0, 'cleanliness_rating': 2.0, 'location_rating': 2.0, 'rooms_rating': 8.0, 'service_rating': 8.0, 'value_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 7.0, 'cleanliness_rating': 5.0, 'location_rating': 5.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 7.0},
    {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 5.0, 'cleanliness_rating': 2.0, 'location_rating': 2.0, 'rooms_rating': 8.0, 'service_rating': 8.0, 'value_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 4, 'overall_rating': 7.0, 'cleanliness_rating': 5.0, 'location_rating': 5.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 7.0},
    # {'user_id': 'U1', 'offering_id': 5, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 1, 'overall_rating': 5.0, 'cleanliness_rating': 8.0, 'location_rating': 8.0, 'rooms_rating': 2.0, 'service_rating': 2.0, 'value_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0, 'cleanliness_rating': 9.0, 'location_rating': 9.0, 'rooms_rating': 5.0, 'service_rating': 5.0, 'value_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 3, 'overall_rating': 5.0, 'cleanliness_rating': 8.0, 'location_rating': 8.0, 'rooms_rating': 2.0, 'service_rating': 2.0, 'value_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 4, 'overall_rating': 7.0, 'cleanliness_rating': 9.0, 'location_rating': 9.0, 'rooms_rating': 5.0, 'service_rating': 5.0, 'value_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 5, 'overall_rating': 9.0, 'cleanliness_rating': 9.0, 'location_rating': 9.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 9.0},
    {'user_id': 'U3', 'offering_id': 1, 'overall_rating': 5.0, 'cleanliness_rating': 8.0, 'location_rating': 8.0, 'rooms_rating': 2.0, 'service_rating': 2.0, 'value_rating': 5.0},
    {'user_id': 'U3', 'offering_id': 2, 'overall_rating': 7.0, 'cleanliness_rating': 9.0, 'location_rating': 9.0, 'rooms_rating': 5.0, 'service_rating': 5.0, 'value_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 3, 'overall_rating': 5.0, 'cleanliness_rating': 8.0, 'location_rating': 8.0, 'rooms_rating': 2.0, 'service_rating': 2.0, 'value_rating': 5.0},
    {'user_id': 'U3', 'offering_id': 4, 'overall_rating': 7.0, 'cleanliness_rating': 9.0, 'location_rating': 9.0, 'rooms_rating': 5.0, 'service_rating': 5.0, 'value_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 5, 'overall_rating': 9.0, 'cleanliness_rating': 9.0, 'location_rating': 9.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 9.0},
    {'user_id': 'U4', 'offering_id': 1, 'overall_rating': 6.0, 'cleanliness_rating': 3.0, 'location_rating': 3.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 2, 'overall_rating': 6.0, 'cleanliness_rating': 3.0, 'location_rating': 3.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 3, 'overall_rating': 6.0, 'cleanliness_rating': 4.0, 'location_rating': 4.0, 'rooms_rating': 8.0, 'service_rating': 8.0, 'value_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 4, 'overall_rating': 6.0, 'cleanliness_rating': 4.0, 'location_rating': 4.0, 'rooms_rating': 8.0, 'service_rating': 8.0, 'value_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 5, 'overall_rating': 5.0, 'cleanliness_rating': 5.0, 'location_rating': 5.0, 'rooms_rating': 5.0, 'service_rating': 5.0, 'value_rating': 5.0},
    {'user_id': 'U5', 'offering_id': 1, 'overall_rating': 6.0, 'cleanliness_rating': 3.0, 'location_rating': 3.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 2, 'overall_rating': 6.0, 'cleanliness_rating': 3.0, 'location_rating': 3.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 3, 'overall_rating': 6.0, 'cleanliness_rating': 4.0, 'location_rating': 4.0, 'rooms_rating': 8.0, 'service_rating': 8.0, 'value_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 4, 'overall_rating': 6.0, 'cleanliness_rating': 4.0, 'location_rating': 4.0, 'rooms_rating': 8.0, 'service_rating': 8.0, 'value_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 5, 'overall_rating': 5.0, 'cleanliness_rating': 5.0, 'location_rating': 5.0, 'rooms_rating': 5.0, 'service_rating': 5.0, 'value_rating': 5.0}
]




class TestDeltaCFRecommender(TestCase):

    def test_predict_rating(self):
        recommender = DeltaCFRecommender(significant_criteria_ranges=None)
        recommender.load(reviews_matrix_5)

        # print(recommender.user_similarity_matrix['U1']['U2'])
        # print(recommender.user_similarity_matrix['U1']['U3'])
        # print(recommender.user_similarity_matrix['U1']['U4'])
        # print(recommender.user_similarity_matrix['U1']['U5'])
        # print(recommender.user_dictionary['U2'].average_overall_rating)
        # print(recommender.user_dictionary['U4'].average_overall_rating)

        numerator = (0.291666666667 * (9 - 6.6)) * 2 + (0.142857142857 * (5 - 5.8)) * 2
        denominator = (0.291666666667 * 2) + (0.142857142857 * 2)
        expected_result = 6 + numerator / denominator

        self.assertAlmostEqual(expected_result, recommender.predict_rating('U1', 5), 7)
        self.assertEqual(5.9739130434782615, recommender.predict_rating('U4', 5))
        self.assertEqual(None, recommender.predict_rating('U4', 6))

        recommender.load(reviews_matrix_5[3:])
        self.assertEqual(8.5815111411579519, recommender.predict_rating('U1', 5))
