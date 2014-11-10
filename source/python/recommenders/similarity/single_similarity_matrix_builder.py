from etl import similarity_calculator
from recommenders.similarity.base_similarity_matrix_builder import \
    BaseSimilarityMatrixBuilder
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class SingleSimilarityMatrixBuilder(BaseSimilarityMatrixBuilder):
    def __init__(self, similarity_metric):
        super(SingleSimilarityMatrixBuilder, self).__init__(
            'SingleSimilarity', similarity_metric, False)

    def calculate_users_similarity(self, user_dictionary, user1, user2):
        common_items = extractor.get_common_items(user_dictionary, user1, user2)

        if not common_items:
            return None

        if self._min_common_items is not None and len(
                common_items) < self._min_common_items:
            return None

        user1_ratings =\
            extractor.get_user_ratings(user_dictionary, user1, common_items)
        user2_ratings =\
            extractor.get_user_ratings(user_dictionary, user2, common_items)

        similarity_value = similarity_calculator.calculate_similarity(
            user1_ratings, user2_ratings, self._similarity_metric)

        return similarity_value
