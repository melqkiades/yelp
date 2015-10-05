from recommenders.context.neighbour_contribution.abstract_neighbour_contribution_calculator import \
    AbstractNeighbourContributionCalculator
from topicmodeling.context import context_utils

__author__ = 'fpena'


class ContextNCCalculator(AbstractNeighbourContributionCalculator):

    def __init__(self):
        super(ContextNCCalculator, self).__init__()

    def load(self, user_baseline_calculator):
        self.user_baseline_calculator = user_baseline_calculator

    def calculate_neighbour_contribution(
            self, neighbour_id, item_id, context, threshold):

        neighbour_rating = self.user_baseline_calculator.get_rating_on_context(
            neighbour_id, item_id, context, threshold)
        neighbor_average =\
            self.user_baseline_calculator.calculate_user_baseline(
                neighbour_id, context, threshold)
        neighbour_context =\
            self.user_baseline_calculator.user_dictionary[neighbour_id].item_contexts[item_id]
        context_similarity = context_utils.get_context_similarity(
            context, neighbour_context, self.user_baseline_calculator.topic_indices)

        if not neighbour_rating:
            return None
        if not neighbor_average:
            return None

        return (neighbour_rating - neighbor_average) * context_similarity
