from recommenders.base_recommender import BaseRecommender
from pandas import DataFrame

__author__ = 'fpena'


class AverageRecommender(BaseRecommender):

    def __init__(self):
        super(AverageRecommender, self).__init__('AverageRecommender', None)
        self._rating = None

    def predict_rating(self, user_id, hotel_id):
        return self._rating

    def load(self, reviews):
        data_frame = DataFrame(reviews)
        mean = data_frame.mean()['overall_rating']
        self._rating = mean
        print('Mean rating:', self._rating  )
