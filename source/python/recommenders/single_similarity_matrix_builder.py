from etl import similarity_calculator
from recommenders.similarity.base_similarity_matrix_builder import \
    BaseSimilarityMatrixBuilder

__author__ = 'fpena'


class SingleSimilarityMatrixBuilder(BaseSimilarityMatrixBuilder):

    def __init__(self, similarity_metric):
        super(SingleSimilarityMatrixBuilder, self).__init__(
            similarity_metric)

    def calculate_users_similarity(self, user_dictionary, user1, user2):
        common_items = self.get_common_items(user_dictionary, user1, user2)

        if not common_items:
            return None

        user1_ratings =\
            self.extract_user_ratings(user_dictionary, user1, common_items)
        user2_ratings =\
            self.extract_user_ratings(user_dictionary, user2, common_items)

        similarity_value = similarity_calculator.calculate_similarity(
            user1_ratings, user2_ratings, self._similarity_metric)

        return similarity_value

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
            ratings.append(SingleSimilarityMatrixBuilder.get_rating(
                user_dictionary, user, item))

        return ratings

    @staticmethod
    def get_rating(user_dictionary, user, item):
        if item in user_dictionary[user].item_ratings:
            return user_dictionary[user].item_ratings[item]
        return None
