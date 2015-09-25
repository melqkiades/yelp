from recommenders.context.baseline.abstract_user_baseline_calculator import \
    AbstractUserBaselineCalculator

__author__ = 'fpena'


class SimpleUserBaselineCalculator(AbstractUserBaselineCalculator):

    def __init__(self):
        super(SimpleUserBaselineCalculator, self).__init__()
        self.user_dictionary = None
        self.topic_indices = None

    def load(self, user_dictionary, topic_indices):
        self.user_dictionary = user_dictionary
        self.topic_indices = topic_indices

    def calculate_user_baseline(self, user_id, context, threshold):
        user_rated_items = self.user_dictionary[user_id].item_ratings
        num_rated_items = len(user_rated_items)

        ratings_sum = 0.0
        num_ratings = 0.0
        for item_id in user_rated_items.keys():
            rating = self.user_dictionary[user_id].item_ratings[item_id]
            ratings_sum += rating
            num_ratings += 1

        user_baseline = ratings_sum / num_rated_items
        return user_baseline

    def get_rating_on_context(self, user_id, item_id, context, threshold):
        return self.user_dictionary[user_id].item_ratings[item_id]
