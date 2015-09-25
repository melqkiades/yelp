from recommenders.context.neighbour_contribution.abstract_neighbour_contribution_calculator import \
    AbstractNeighbourContributionCalculator

__author__ = 'fpena'


class NeighbourContributionCalculator(AbstractNeighbourContributionCalculator):

    def __init__(self):
        super(NeighbourContributionCalculator, self).__init__()

    def load(self, user_baseline_calculator):
        self.user_baseline_calculator = user_baseline_calculator

    def calculate_neighbour_contribution(
            self, neighbour_id, item_id, context, threshold):

        neighbour_rating = self.user_baseline_calculator.get_rating_on_context(
            neighbour_id, item_id, context, threshold)
        neighbor_average =\
            self.user_baseline_calculator.calculate_user_baseline(
                neighbour_id, context, threshold)

        if not neighbour_rating:
            return None
        if not neighbor_average:
            return None

        return neighbour_rating - neighbor_average
