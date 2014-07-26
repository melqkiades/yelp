from etl import ETLUtils

__author__ = 'fpena'

from unittest import TestCase


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
    {'user_id': 'U2', 'offering_id': 5, 'overall_rating': 9.0, 'cleanliness_rating': 9.0, 'location_rating': 9.0, 'rooms_rating': 9.0, 'service_rating': 9.0, 'value_rating': 9.0}
]

reviews_matrix_5_short = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 4, 'overall_rating': 7.0},
    # {'user_id': 'U1', 'offering_id': 5, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 4, 'overall_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 5, 'overall_rating': 9.0}
]

reviews_matrix_5_users = [
    {'user_id': 'U1'},
    {'user_id': 'U1'},
    {'user_id': 'U1'},
    {'user_id': 'U1'},
    # {'user_id': 'U1', 'offering_id': 5},
    {'user_id': 'U2'},
    {'user_id': 'U2'},
    {'user_id': 'U2'},
    {'user_id': 'U2'},
    {'user_id': 'U2'}
]


class TestETLUtils(TestCase):

    def test_drop_fields(self):

        drop_fields = [
            'cleanliness_rating',
            'location_rating',
            'rooms_rating',
            'service_rating',
            'value_rating'
        ]

        test_list = list(reviews_matrix_5)

        ETLUtils.drop_fields(drop_fields, test_list)
        self.assertEqual(reviews_matrix_5_short, test_list)

        test_list = list(reviews_matrix_5_short)
        self.assertEqual(reviews_matrix_5_short, test_list)

    def test_select_fields(self):

        select_fields = ['user_id', 'offering_id', 'overall_rating']
        result = ETLUtils.select_fields(select_fields, reviews_matrix_5)
        self.assertEqual(result, reviews_matrix_5_short)

        select_fields = ['user_id']
        result = ETLUtils.select_fields(select_fields, reviews_matrix_5_short)
        self.assertEqual(result, reviews_matrix_5_users)

    def test_filter_records(self):

        field = 'offering_id'
        values = [1, 3, 5]

        expected_result = [
            {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0},
            {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 5.0},
            {'user_id': 'U2', 'offering_id': 1, 'overall_rating': 5.0},
            {'user_id': 'U2', 'offering_id': 3, 'overall_rating': 5.0},
            {'user_id': 'U2', 'offering_id': 5, 'overall_rating': 9.0}
        ]

        actual_result = ETLUtils.filter_records(reviews_matrix_5_short, field, values)

        self.assertEqual(expected_result, actual_result)

    def test_filter_out_records(self):

        field = 'offering_id'
        values = [1, 3, 5]

        expected_result = [
            {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 7.0},
            {'user_id': 'U1', 'offering_id': 4, 'overall_rating': 7.0},
            {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0},
            {'user_id': 'U2', 'offering_id': 4, 'overall_rating': 7.0}
        ]

        actual_result = ETLUtils.filter_out_records(reviews_matrix_5_short, field, values)

        self.assertEqual(expected_result, actual_result)

    # def
