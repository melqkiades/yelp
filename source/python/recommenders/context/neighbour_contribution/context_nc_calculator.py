from topicmodeling.context import context_utils

__author__ = 'fpena'


class ContextNCCalculator:

    def __init__(self):
        self.ubc = None

    def load(self, user_baseline_calculator):
        self.ubc = user_baseline_calculator

    def calculate_neighbour_contribution(
            self, neighbour_id, item_id, context, threshold):

        neighbour_rating = self.ubc.get_rating_on_context(
            neighbour_id, item_id, context, threshold)
        neighbor_average =\
            self.ubc.calculate_user_baseline(
                neighbour_id, context, threshold)
        neighbour_context =\
            self.ubc.user_dictionary[neighbour_id].item_contexts[item_id]
        context_similarity = context_utils.get_context_similarity(
            context, neighbour_context, self.ubc.topic_indices)

        return (neighbour_rating - neighbor_average) * context_similarity
