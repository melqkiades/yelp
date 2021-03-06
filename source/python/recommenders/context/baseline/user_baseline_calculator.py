from recommenders.context.baseline.abstract_user_baseline_calculator import \
    AbstractUserBaselineCalculator
from topicmodeling.context import context_utils

__author__ = 'fpena'


class UserBaselineCalculator(AbstractUserBaselineCalculator):

    def __init__(self):
        super(UserBaselineCalculator, self).__init__()
        self.user_dictionary = None
        self.topic_indices = None

    def load(self, user_dictionary, topic_indices):
        self.user_dictionary = user_dictionary
        self.topic_indices = topic_indices

    def calculate_user_baseline(self, user_id, context, threshold):
        user_rated_items = self.user_dictionary[user_id].item_ratings

        ratings_sum = 0.0
        num_ratings = 0.0
        for item_id in user_rated_items.keys():
            rating = self.get_rating_on_context(
                user_id, item_id, context, threshold)
            if rating:
                ratings_sum += rating
                num_ratings += 1

        if num_ratings == 0:
            return None

        user_baseline = ratings_sum / num_ratings

        return user_baseline

    def get_rating_on_context(self, user, item, context, threshold):

        neighbour_context = self.user_dictionary[user].item_contexts[item]
        context_similarity = context_utils.get_context_similarity(
            context, neighbour_context, self.topic_indices)

        if context_similarity < threshold:
            return None

        return self.user_dictionary[user].item_ratings[item]
