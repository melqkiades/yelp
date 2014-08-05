from abc import abstractmethod, ABCMeta

__author__ = 'fpena'


class BaseSimilarityMatrixBuilder:

    __metaclass__ = ABCMeta

    def __init__(self, similarity_metric):
        self._similarity_metric = similarity_metric

    def build_similarity_matrix(self, user_dictionary, user_ids):
        """
        Builds a matrix that contains the similarity between every pair of users
        in the dataset of this recommender system. This is particularly useful
        to prevent repeating the same calculations in each cycle

        """
        user_similarity_matrix = {}

        for user1 in user_ids:
            user_similarity_matrix[user1] = {}
            for user2 in user_ids:
                user_similarity_matrix[user1][user2] =\
                    self.calculate_users_similarity(
                        user_dictionary, user1, user2)

        return user_similarity_matrix

    @abstractmethod
    def calculate_users_similarity(self, user_dictionary, user_id1, user_id2):
        pass
