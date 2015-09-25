from recommenders.context.neighbourhood.abstract_neighbourhood_calculator import \
    AbstractNeighbourhoodCalculator
from utils import dictionary_utils

__author__ = 'fpena'


class SimpleNeighbourhoodCalculator(AbstractNeighbourhoodCalculator):

    def __init__(self, user_similarity_calculator):
        super(SimpleNeighbourhoodCalculator, self).__init__()
        self.user_ids = None
        self.user_dictionary = None
        self.topic_indices = None
        self.num_neighbours = None
        self.similarity_matrix = None
        self.user_similarity_calculator = user_similarity_calculator

    def load(self, user_ids, user_dictionary, topic_indices, num_neighbours):
        self.user_ids = user_ids
        self.user_dictionary = user_dictionary
        self.topic_indices = topic_indices
        self.num_neighbours = num_neighbours
        self.user_similarity_calculator.load(
            self.user_ids, self.user_dictionary, self.topic_indices)
        self.similarity_matrix =\
            self.user_similarity_calculator.create_similarity_matrix()

    def get_neighbourhood(self, user, item, context, threshold):

        sim_users_matrix = self.similarity_matrix[user].copy()
        sim_users_matrix.pop(user, None)

        # We remove the users who have not rated the given item
        sim_users_matrix = {
            k: v for k, v in sim_users_matrix.items()
            if item in self.user_dictionary[k].item_ratings}

        # We remove neighbours that don't have a similarity with user
        sim_users_matrix = {
            k: v for k, v in sim_users_matrix.items()
            if v}

        # Sort the users by similarity
        neighbourhood = dictionary_utils.sort_dictionary_keys(
            sim_users_matrix)

        if self.num_neighbours is None:
            return neighbourhood

        return neighbourhood[:self.num_neighbours]

    def clear(self):
        self.user_ids = None
        self.user_dictionary = None
        self.topic_indices = None
        self.similarity_matrix = None
        self.user_similarity_calculator.clear()
