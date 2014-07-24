from tripadvisor.fourcity.reviews_holder import ReviewsHolder

__author__ = 'fpena'


class BaseRecommender(object):

    def __init__(self, name):
        self._name = name
        self.reviews = None
        self.reviews_holder = None
        self.user_ids = None

    def load(self, reviews):
        self.reviews = reviews
        self.reviews_holder = ReviewsHolder(self.reviews)
        self.user_ids = self.reviews_holder.user_ids

    def clear(self):
        self.reviews = None
        self.reviews_holder = None
        self.user_ids = None

    @property
    def name(self):
        return self._name
