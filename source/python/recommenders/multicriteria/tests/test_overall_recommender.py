from unittest import TestCase
from recommenders.multicriteria.overall_recommender import OverallRecommender

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




class TestOverallRecommender(TestCase):

    def test_predict_rating(self):
        recommender = OverallRecommender(significant_criteria_ranges=None)
        recommender.load(reviews_matrix_5)
        self.assertEqual(7, recommender.predict_rating('U1', 5))
        self.assertEqual(7.666666666666667, recommender.predict_rating('U4', 5))
        self.assertEqual(None, recommender.predict_rating('U4', 6))

        recommender.load(reviews_matrix_5[3:])
        self.assertEqual(7, recommender.predict_rating('U1', 5))
