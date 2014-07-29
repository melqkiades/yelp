from recommenders.base_recommender import BaseRecommender

__author__ = 'fpena'


class DummyRecommender(BaseRecommender):

    def __init__(self, rating):
        super(DummyRecommender, self).__init__('DummyRecommender', None)
        self._rating = rating

    def predict_rating(self, user_id, hotel_id):
        return self._rating

    def load(self, reviews):
        pass
