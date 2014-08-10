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

        common_items = self.get_common_items(user_dictionary, user1, user2)

        if not common_items:
            return None

        user1_overall_ratings =\
            self.extract_user_ratings(user_dictionary, user1, common_items)
        user1_multi_ratings =\
            self.extract_user_multi_ratings(user_dictionary, user1, common_items)

        user2_overall_ratings =\
            self.extract_user_ratings(user_dictionary, user2, common_items)
        user2_multi_ratings =\
            self.extract_user_multi_ratings(user_dictionary, user2, common_items)

        num_criteria = len(user1_multi_ratings[0])
        total_similarity = 0.

        for i in xrange(0, num_criteria):
            user1_criterion_item_ratings = self.column(user1_multi_ratings, i)
            user2_criterion_item_ratings = self.column(user2_multi_ratings, i)

            total_similarity += similarity_calculator.calculate_similarity(
                user1_criterion_item_ratings, user2_criterion_item_ratings,
                self._similarity_metric)

        # We also add the overall similarity
        total_similarity += similarity_calculator.calculate_similarity(
            user1_overall_ratings, user2_overall_ratings, self._similarity_metric)

        average_similarity = total_similarity / (num_criteria + 1)

        return average_similarity

    @staticmethod
    def get_common_items(user_dictionary, user1, user2):
        items_user1 = set(user_dictionary[user1].item_ratings.keys())
        items_user2 = set(user_dictionary[user2].item_ratings.keys())

        common_items = items_user1.intersection(items_user2)

        return common_items

    @staticmethod
    def extract_user_ratings(user_dictionary, user, items):

        ratings = []

        for item in items:
            ratings.append(AverageSimilarityMatrixBuilder.get_rating(
                user_dictionary, user, item))

        return ratings

    @staticmethod
    def get_rating(user_dictionary, user, item):
        if item in user_dictionary[user].item_ratings:
            return user_dictionary[user].item_ratings[item]
        return None

    @staticmethod
    def extract_user_multi_ratings(user_dictionary, user, items):

        ratings = []

        for item in items:
            ratings.append(AverageSimilarityMatrixBuilder.get_multi_ratings(
                user_dictionary, user, item))

        return ratings


    @staticmethod
    def get_multi_ratings(user_dictionary, user, item):
        if item in user_dictionary[user].item_multi_ratings:
            return user_dictionary[user].item_multi_ratings[item]
        return None

    @staticmethod
    def column(matrix, i):
        return [row[i] for row in matrix]




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
