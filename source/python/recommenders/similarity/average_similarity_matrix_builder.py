from etl import similarity_calculator
from recommenders.similarity.base_similarity_matrix_builder import \
    BaseSimilarityMatrixBuilder
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class AverageSimilarityMatrixBuilder(BaseSimilarityMatrixBuilder):

    def __init__(self, similarity_metric):
        super(AverageSimilarityMatrixBuilder, self).__init__(
            'MultiAverageSimilarity', similarity_metric, True)

    def calculate_users_similarity(self, user_dictionary, user1, user2):

        common_items = extractor.get_common_items(user_dictionary, user1, user2)

        if not common_items:
            return None

        if self._min_common_items is not None and len(
                common_items) < self._min_common_items:
            return None

        user1_overall_ratings =\
            extractor.get_user_ratings(user_dictionary, user1, common_items)
        user1_multi_ratings =\
            extractor.get_user_multi_ratings(user_dictionary, user1, common_items)

        user2_overall_ratings =\
            extractor.get_user_ratings(user_dictionary, user2, common_items)
        user2_multi_ratings =\
            extractor.get_user_multi_ratings(user_dictionary, user2, common_items)

        num_criteria = len(user1_multi_ratings[0])
        total_similarity = 0.

        for i in xrange(0, num_criteria):
            user1_criterion_item_ratings =\
                extractor.get_matrix_column(user1_multi_ratings, i)
            user2_criterion_item_ratings =\
                extractor.get_matrix_column(user2_multi_ratings, i)

            total_similarity += similarity_calculator.calculate_similarity(
                user1_criterion_item_ratings, user2_criterion_item_ratings,
                self._similarity_metric)

        # We also add the overall similarity
        total_similarity += similarity_calculator.calculate_similarity(
            user1_overall_ratings, user2_overall_ratings, self._similarity_metric)

        average_similarity = total_similarity / (num_criteria + 1)

        return average_similarity




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

# user_dict = extractor.initialize_users(reviews_matrix_5_1, True)
# similarity_matrix_builder = AverageSimilarityMatrixBuilder('euclidean')
#
# similarity_matrix_builder.calculate_users_similarity(user_dict, 'U1', 'U2')
