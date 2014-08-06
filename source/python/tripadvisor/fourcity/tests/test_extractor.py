from unittest import TestCase
from tripadvisor.fourcity import extractor

__author__ = 'fpena'



reviews_matrix_5_1 = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 4.0, 'multi_ratings': [3.0, 4.0, 10.0, 7.0, 5.0]},
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


user_item_multi_ratings = {
    1: [2.5, 3.0, 9.0, 7.5, 5.0],
    2: [5.0, 5.0, 9.0, 9.0, 7.0],
    3: [2.0, 2.0, 8.0, 8.0, 5.0],
    4: [5.0, 5.0, 9.0, 9.0, 7.0]
}


class TestExtractor(TestCase):

    def test_get_user_multi_item_ratings(self):

        actual_value = extractor.get_user_item_multi_ratings(
            reviews_matrix_5_1, 'U1', True)

        self.assertEqual(user_item_multi_ratings, actual_value)
