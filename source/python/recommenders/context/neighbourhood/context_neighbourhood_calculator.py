from utils import dictionary_utils
from topicmodeling.context import context_utils

__author__ = 'fpena'


class ContextNeighbourhoodCalculator:

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

        all_users = self.user_ids[:]
        all_users.remove(user)
        neighbours = []

        # We remove the users who have not rated the given item
        for neighbour in all_users:
            if item in self.user_dictionary[neighbour].item_ratings:
                neighbours.append(neighbour)

        neighbour_similarity_map = {}
        for neighbour in neighbours:
            neighbour_context =\
                self.user_dictionary[neighbour].item_contexts[item]
            context_similarity = context_utils.get_context_similarity(
                context, neighbour_context, self.topic_indices)
            if context_similarity > threshold:
                neighbour_similarity_map[neighbour] = context_similarity

        # Sort the users by similarity
        neighbourhood = dictionary_utils.sort_dictionary_keys(
            neighbour_similarity_map)  # [:self.num_neighbors]

        return neighbourhood
