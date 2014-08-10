from abc import ABCMeta

from recommenders.similarity.weights_similarity_matrix_builder import \
    WeightsSimilarityMatrixBuilder
from tripadvisor.fourcity import extractor
from recommenders.base_recommender import BaseRecommender


__author__ = 'fpena'


class MultiCriteriaBaseRecommender(BaseRecommender):

    __metaclass__ = ABCMeta

    def __init__(
            self, name, similarity_metric=None,
            significant_criteria_ranges=None):
        super(MultiCriteriaBaseRecommender, self).__init__(name, None)
        self._significant_criteria_ranges = significant_criteria_ranges
        self._similarity_matrix_builder = WeightsSimilarityMatrixBuilder(similarity_metric)
        self.user_cluster_dictionary = None

    def load(self, reviews):
        self.reviews = reviews
        self.user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        self.user_dictionary =\
            extractor.initialize_cluster_users(self.reviews, self._significant_criteria_ranges)
        self.user_cluster_dictionary = self.build_user_clusters(
            self.reviews, self._significant_criteria_ranges)
        if self._similarity_matrix_builder._similarity_metric is not None:
            self.user_similarity_matrix =\
                self._similarity_matrix_builder.build_similarity_matrix(
                    self.user_dictionary, self.user_ids)

    def clear(self):
        super(MultiCriteriaBaseRecommender, self).clear()
        self.user_cluster_dictionary = None

    @staticmethod
    def build_user_clusters(reviews, significant_criteria_ranges=None):
        """
        Builds a series of clusters for users according to their significant
        criteria. Users that have exactly the same significant criteria will belong
        to the same cluster.

        :param reviews: the list of reviews
        :return: a dictionary where all the keys are the cluster names and the
        values for those keys are list of users that belong to that cluster
        """

        user_list = extractor.get_groupby_list(reviews, 'user_id')
        user_cluster_dictionary = {}

        for user in user_list:
            weights = extractor.get_criteria_weights(reviews, user)
            significant_criteria, cluster_name =\
                extractor.get_significant_criteria(weights, significant_criteria_ranges)

            if cluster_name in user_cluster_dictionary:
                user_cluster_dictionary[cluster_name].append(user)
            else:
                user_cluster_dictionary[cluster_name] = [user]

        return user_cluster_dictionary
