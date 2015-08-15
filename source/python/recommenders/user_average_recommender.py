from recommenders.base_recommender import BaseRecommender
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class UserAverageRecommender(BaseRecommender):

    def __init__(self):
        super(UserAverageRecommender, self).__init__('AverageRecommender', None)
        self._rating = None

    def load(self, reviews):
        self.reviews = reviews
        self.user_dictionary = extractor.initialize_users(self.reviews, False)
        self.user_ids = extractor.get_groupby_list(self.reviews, 'user_id')

    def predict_rating(self, user_id, item_id):
        return self.user_dictionary[user_id].average_overall_rating
