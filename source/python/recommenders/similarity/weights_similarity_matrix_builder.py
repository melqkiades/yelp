from etl import similarity_calculator
from recommenders.similarity.base_similarity_matrix_builder import \
    BaseSimilarityMatrixBuilder

__author__ = 'fpena'


class WeightsSimilarityMatrixBuilder(BaseSimilarityMatrixBuilder):

    def __init__(self, similarity_metric):
        super(WeightsSimilarityMatrixBuilder, self).__init__(
            similarity_metric)

    def calculate_users_similarity(self, user_dictionary, user_id1, user_id2):
        """
        Calculates the similarity between two users based on how similar are
        their ratings in the reviews

        :param user_id1: the ID of user 1
        :param user_id2: the ID of user 2
        :return: a float with the similarity between the two users. Since this
        function is based on euclidean distance to calculate the similarity, a
        similarity of 0 indicates that the users share exactly the same tastes
        """
        user_weights1 = user_dictionary[user_id1].criteria_weights
        user_weights2 = user_dictionary[user_id2].criteria_weights

        return similarity_calculator.calculate_similarity(
            user_weights1, user_weights2, self._similarity_metric)

