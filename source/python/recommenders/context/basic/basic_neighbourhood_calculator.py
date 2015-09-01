from utils import dictionary_utils

__author__ = 'fpena'


class BasicNeighbourhoodCalculator:

    def __init__(self):
        self.user_ids = None
        self.user_dictionary = None
        self.topic_indices = None
        self.num_neighbours = None
        self.user_similarity_matrix = None

    def load(self, user_ids, user_dictionary,
             topic_indices, num_neighbours, user_similarity_matrix):
        self.user_ids = user_ids
        self.user_dictionary = user_dictionary
        self.topic_indices = topic_indices
        self.num_neighbours = num_neighbours
        self.user_similarity_matrix = user_similarity_matrix

    def get_neighbourhood(self, user, item, context, threshold):

        sim_users_matrix = self.user_similarity_matrix[user].copy()
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

        if not self.num_neighbours:
            return neighbourhood

        return neighbourhood[:self.num_neighbours]
