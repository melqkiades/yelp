from unittest import TestCase

from recommenders.similarity.single_similarity_matrix_builder import \
    SingleSimilarityMatrixBuilder
from tripadvisor.fourcity import extractor


__author__ = 'fpena'


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

euclidean_matrix = {
    'U1': {'U1': 1.0, 'U2': 1.0, 'U3': 1.0, 'U4': 0.36602540378443865, 'U5': 0.36602540378443865},
    'U2': {'U1': 1.0, 'U2': 1.0, 'U3': 1.0, 'U4': 0.1827439976315568, 'U5': 0.1827439976315568},
    'U3': {'U1': 1.0, 'U2': 1.0, 'U3': 1.0, 'U4': 0.1827439976315568, 'U5': 0.1827439976315568},
    'U4': {'U1': 0.36602540378443865, 'U2': 0.1827439976315568, 'U3': 0.1827439976315568, 'U4': 1.0, 'U5': 1.0},
    'U5': {'U1': 0.36602540378443865, 'U2': 0.1827439976315568, 'U3': 0.1827439976315568, 'U4': 1.0, 'U5': 1.0}
}

cosine_matrix = {
    'U1': {'U5': 0.98910049196117167, 'U4': 0.98910049196117167, 'U1': 0.99999999999999989, 'U3': 0.99999999999999989, 'U2': 0.99999999999999989},
    'U2': {'U5': 0.96072858066163047, 'U4': 0.96072858066163047, 'U1': 0.99999999999999989, 'U3': 0.99999999999999989, 'U2': 0.99999999999999989},
    'U3': {'U5': 0.96072858066163047, 'U4': 0.96072858066163047, 'U1': 0.99999999999999989, 'U3': 0.99999999999999989, 'U2': 0.99999999999999989},
    'U4': {'U5': 1.0, 'U4': 1.0, 'U1': 0.98910049196117167, 'U3': 0.96072858066163047, 'U2': 0.96072858066163047},
    'U5': {'U5': 1.0, 'U4': 1.0, 'U1': 0.98910049196117167, 'U3': 0.96072858066163047, 'U2': 0.96072858066163047}
}

pearson_matrix = {
    'U1': {'U1': 1.0, 'U3': 1.0, 'U2': 1.0},
    'U2': {'U1': 1.0, 'U3': 1.0, 'U2': 1.0},
    'U3': {'U1': 1.0, 'U3': 1.0, 'U2': 1.0},
    'U4': {'U5': 1.0, 'U4': 1.0},
    'U5': {'U5': 1.0, 'U4': 1.0}
}


class TestSimilarityMatrixBuilder(TestCase):

    def test_build_similarity_matrix_euclidean(self):

        user_dictionary =\
            extractor.initialize_users(reviews_matrix_5, False)
        user_ids = extractor.get_groupby_list(reviews_matrix_5, 'user_id')
        similarity_matrix_builder = SingleSimilarityMatrixBuilder('euclidean')
        self.assertEqual(euclidean_matrix, similarity_matrix_builder.build_similarity_matrix(
            user_dictionary, user_ids))

    def test_build_similarity_matrix_cosine(self):

        user_dictionary =\
            extractor.initialize_users(reviews_matrix_5, False)
        user_ids = extractor.get_groupby_list(reviews_matrix_5, 'user_id')
        similarity_matrix_builder = SingleSimilarityMatrixBuilder('cosine')
        self.assertEqual(cosine_matrix, similarity_matrix_builder.build_similarity_matrix(
            user_dictionary, user_ids))

    def test_build_similarity_matrix_pearson(self):

        user_dictionary =\
            extractor.initialize_users(reviews_matrix_5, False)
        user_ids = extractor.get_groupby_list(reviews_matrix_5, 'user_id')
        similarity_matrix_builder = SingleSimilarityMatrixBuilder('pearson')
        self.assertEqual(pearson_matrix, similarity_matrix_builder.build_similarity_matrix(
            user_dictionary, user_ids))

