from abc import ABCMeta

from tripadvisor.fourcity import extractor
from tripadvisor.fourcity import fourcity_clusterer
from recommenders.base_recommender import BaseRecommender


__author__ = 'fpena'


class MultiCriteriaBaseRecommender(BaseRecommender):

    __metaclass__ = ABCMeta

    def __init__(
            self, name, similarity_metric=None,
            significant_criteria_ranges=None):
        super(MultiCriteriaBaseRecommender, self).__init__(name, similarity_metric)
        self._significant_criteria_ranges = significant_criteria_ranges
        self.user_cluster_dictionary = None

    def load(self, reviews):
        super(MultiCriteriaBaseRecommender, self).load(reviews)
        self.user_dictionary =\
            extractor.initialize_cluster_users(self.reviews, self._significant_criteria_ranges)
        self.user_cluster_dictionary = fourcity_clusterer.build_user_clusters(
            self.reviews, self._significant_criteria_ranges)
        if self._similarity_metric is not None:
            self.user_similarity_matrix =\
                fourcity_clusterer.build_user_similarities_matrix(
                    self.user_ids, self.user_dictionary, self._similarity_metric)

    def clear(self):
        super(MultiCriteriaBaseRecommender, self).clear()
        self.user_cluster_dictionary = None
