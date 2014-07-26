from unittest import TestCase
from tripadvisor.fourcity.reviews_holder import ReviewsHolder

__author__ = 'fpena'


reviews = [
    {
        'user_id': 'A1',
        'offering_id': 1,
        'overall_rating': 4.0,
        'location_rating': 5.0,
        'cleanliness_rating': 5.0,
        'rooms_rating': 5.0,
        'service_rating': 4.0,
        'value_rating': 3.0
    },
    {
        'user_id': 'A1',
        'offering_id': 1,
        'overall_rating': 5.0,
        'location_rating': 5.0,
        'cleanliness_rating': 5.0,
        'rooms_rating': 5.0,
        'service_rating': 4.0,
        'value_rating': 3.0
    },
    {
        'user_id': 'A1',
        'offering_id': 2,
        'overall_rating': 2.0,
        'location_rating': 5.0,
        'cleanliness_rating': 5.0,
        'rooms_rating': 5.0,
        'service_rating': 4.0,
        'value_rating': 3.0
    },
    {
        'user_id': 'A1',
        'offering_id': 3,
        'overall_rating': 3.0,
        'location_rating': 5.0,
        'cleanliness_rating': 5.0,
        'rooms_rating': 5.0,
        'service_rating': 4.0,
        'value_rating': 3.0
    },
    {
        'user_id': 'A2',
        'offering_id': 1,
        'overall_rating': 4.0,
        'location_rating': 5.0,
        'cleanliness_rating': 5.0,
        'rooms_rating': 5.0,
        'service_rating': 4.0,
        'value_rating': 3.0
    },
    {
        'user_id': 'A2',
        'offering_id': 3,
        'overall_rating': 2.0,
        'location_rating': 5.0,
        'cleanliness_rating': 5.0,
        'rooms_rating': 5.0,
        'service_rating': 4.0,
        'value_rating': 3.0
    },
]


class TestReviewsHolder(TestCase):

    def test_get_rating(self):

        reviews_holder = ReviewsHolder(reviews)

        self.assertEqual(4.5, reviews_holder.get_rating('A1', 1))
        self.assertEqual(2.0, reviews_holder.get_rating('A1', 2))
        # self.fail()

    def test_get_user_average_rating(self):

        reviews_holder = ReviewsHolder(reviews)

        self.assertEqual(3.5, reviews_holder.get_user_average_rating('A1'))

    def test_get_common_items(self):

        reviews_holder = ReviewsHolder(reviews)
        expected_items = {1, 3}

        self.assertEqual(expected_items, reviews_holder.get_common_items('A1', 'A2'))

    def test_extract_user_ratings(self):

        reviews_holder = ReviewsHolder(reviews)
        items = [1, 2, 3]

        expected_ratings = [4.5, 2, 3]
        self.assertEqual(expected_ratings, reviews_holder.extract_user_ratings('A1', items))

        expected_ratings = [4, None, 2]
        self.assertEqual(expected_ratings, reviews_holder.extract_user_ratings('A2', items))