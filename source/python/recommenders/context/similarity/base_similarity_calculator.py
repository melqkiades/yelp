from abc import abstractmethod, ABCMeta
import itertools

__author__ = 'fpena'


class BaseSimilarityCalculator:

    __metaclass__ = ABCMeta

    def __init__(self):
        self.user_ids = None
        self.user_dictionary = None
        self.context_rich_topics = None

    def create_similarity_matrix(self):
        """
        Builds a matrix that contains the similarity between every pair of users
        in the dataset of this recommender system. This is particularly useful
        to prevent repeating the same calculations in each cycle

        """
        similarity_matrix = {}

        for user in self.user_ids:
            similarity_matrix[user] = {}

        for user_id1, user_id2 in itertools.combinations(self.user_ids, 2):
            similarity =\
                self.calculate_user_similarity(user_id1, user_id2, 0.0)
            similarity_matrix[user_id1][user_id2] = similarity
            similarity_matrix[user_id2][user_id1] = similarity

        return similarity_matrix

    def load(self, user_ids, user_dictionary, context_rich_topics):
        self.user_ids = user_ids
        self.user_dictionary = user_dictionary
        self.context_rich_topics = context_rich_topics

    @abstractmethod
    def calculate_user_similarity(self, user_id1, user_id2, threshold):
        """
        Calculates the similarity between two users
        """

