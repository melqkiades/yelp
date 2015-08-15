from recommenders.base_recommender import BaseRecommender
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class ItemAverageRecommender(BaseRecommender):

    def __init__(self):
        super(ItemAverageRecommender, self).__init__('AverageRecommender', None)
        self._rating = None

    def load(self, reviews):
        self.reviews = reviews

    def predict_rating(self, user_id, item_id):
        return extractor.get_item_average_overall_rating(self.reviews, item_id)
