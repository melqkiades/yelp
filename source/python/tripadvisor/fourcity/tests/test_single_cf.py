from scipy import spatial
from unittest import TestCase
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

    def test_calculate_similarity(self):

        single_cf = SingleCF()
        single_cf.load(reviews)
        similarity = (4.5 * 4 + 3 * 2) / ((4.5**2 + 3**2)**0.5 * (4**2 + 2**2)**0.5)
        self.assertEqual(similarity, single_cf.calculate_similarity('A1', 'A2'))

    def test_calculate_adjusted_weighted_sum(self):

        single_cf = SingleCF()
        single_cf.load(reviews)

        actual_rating_1 = 3.5 + 0.99227787671366752 * (4.0 - 3.0) / 0.99227787671366752
        self.assertEqual(actual_rating_1, single_cf.calculate_adjusted_weighted_sum('A1', 1))
        actual_rating_2 = 3.5 + 0.99227787671366752 * (2.0 - 3.0) / 0.99227787671366752
        self.assertEqual(actual_rating_2, single_cf.calculate_adjusted_weighted_sum('A1', 3))

        single_cf.load(reviews_matrix_5)
        print('Adjusted weighted sum', single_cf.calculate_adjusted_weighted_sum('U1', 5))
        print('Adjusted weighted sum', single_cf.calculate_adjusted_weighted_sum('U1', 1))


    def test_calculate_weighted_sum(self):

        single_cf = SingleCF()
        single_cf.load(reviews)

        actual_rating_1 = 4.0
        self.assertEqual(actual_rating_1, single_cf.calculate_weighted_sum('A1', 1))
        actual_rating_2 = 2.0
        self.assertEqual(actual_rating_2, single_cf.calculate_weighted_sum('A1', 3))
        # print(single_cf.calculate_weighted_sum('A1', 3))

        single_cf.load(reviews_matrix_5)
        print('Weighted sum', single_cf.calculate_weighted_sum('U1', 5))
        print('Weighted sum', single_cf.calculate_weighted_sum('U1', 1))

