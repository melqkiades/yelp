from etl import similarity_calculator
from recommenders.similarity.base_similarity_matrix_builder import \
    BaseSimilarityMatrixBuilder
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class MultiSimilarityMatrixBuilder(BaseSimilarityMatrixBuilder):

    def __init__(self, similarity_metric):
        super(MultiSimilarityMatrixBuilder, self).__init__(
            'MultiStandardSimilarity', similarity_metric, True)

    def calculate_users_similarity(self, user_dictionary, user1, user2):

        common_items = extractor.get_common_items(user_dictionary, user1, user2)

        if not common_items:
            return None

        if self._min_common_items is not None and len(
                common_items) < self._min_common_items:
            return None

        user1_overall_ratings = user_dictionary[user1].item_ratings
        user1_multi_ratings = user_dictionary[user1].item_multi_ratings

        user2_overall_ratings = user_dictionary[user1].item_ratings
        user2_multi_ratings = user_dictionary[user2].item_multi_ratings

        similarity_sum = 0.

        for item in common_items:
            user1_item_ratings = list(user1_multi_ratings[item])
            user1_item_ratings.insert(0, user1_overall_ratings[item])
            user2_item_ratings = list(user2_multi_ratings[item])
            user2_item_ratings.insert(0, user2_overall_ratings[item])

            similarity_sum += similarity_calculator.calculate_similarity(
                user1_item_ratings, user2_item_ratings, self._similarity_metric)

        similarity = similarity_sum / len(common_items)

        return similarity
