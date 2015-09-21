from utils import dictionary_utils
from topicmodeling.context import context_utils

__author__ = 'fpena'


class ContextHybridNeighbourhoodCalculator:

    def __init__(self, user_similarity_calculator, weight=0.5):
        self.user_ids = None
        self.user_dictionary = None
        self.topic_indices = None
        self.num_neighbours = None
        self.weight = weight
        self.similarity_matrix = None
        self.user_similarity_calculator = user_similarity_calculator

    def load(self, user_ids, user_dictionary,
             topic_indices, num_neighbours):
        self.user_ids = user_ids
        self.user_dictionary = user_dictionary
        self.topic_indices = topic_indices
        self.num_neighbours = num_neighbours
        self.user_similarity_calculator.load(
            self.user_ids, self.user_dictionary, self.topic_indices)
        self.similarity_matrix =\
            self.user_similarity_calculator.create_similarity_matrix()

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
            user_similarity = self.similarity_matrix[user][neighbour]
            if context_similarity > threshold and user_similarity:
                neighbour_similarity_map[neighbour] =\
                    self.weight * user_similarity +\
                    (1 - self.weight) * context_similarity

        # Sort the users by similarity
        neighbourhood = dictionary_utils.sort_dictionary_keys(
            neighbour_similarity_map)  # [:self.num_neighbors]

        if self.num_neighbours is None:
            return neighbourhood

        return neighbourhood[:self.num_neighbours]
