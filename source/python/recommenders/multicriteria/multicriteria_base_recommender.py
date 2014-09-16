from abc import ABCMeta

from recommenders.similarity.weights_similarity_matrix_builder import \
    WeightsSimilarityMatrixBuilder
from tripadvisor.fourcity import extractor
from recommenders.base_recommender import BaseRecommender
from utils import dictionary_utils


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

    # TODO: Add the item_id as a parameter in order to optimize the method
    def get_neighbourhood(self, user_id):

        cluster_name = self.user_dictionary[user_id].cluster
        cluster_users = list(self.user_cluster_dictionary[cluster_name])
        cluster_users.remove(user_id)

        # We remove the given user from the cluster in order to avoid bias
        if self._num_neighbors is None:
            return cluster_users

        similarity_matrix = self.user_similarity_matrix[user_id].copy()
        similarity_matrix.pop(user_id, None)
        ordered_similar_users = dictionary_utils.sort_dictionary_keys(
            similarity_matrix)

        intersection_set = set.intersection(set(ordered_similar_users), set(cluster_users))
        intersection_lst = [t for t in ordered_similar_users if t in intersection_set]

        return intersection_lst  # [:self._num_neighbors]

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
