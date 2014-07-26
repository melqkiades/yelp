from recommenders.multicriteria.multicriteria_base_recommender import \
    MultiCriteriaBaseRecommender

__author__ = 'fpena'


class OverallCFRecommender(MultiCriteriaBaseRecommender):

    def __init__(
            self, similarity_metric='euclidean', significant_criteria_ranges=None):
        super(OverallCFRecommender, self).__init__(
            'OverallCFRecommender',
            similarity_metric=similarity_metric,
            significant_criteria_ranges=significant_criteria_ranges)

    def predict_rating(self, user_id, item_id):

        if user_id not in self.user_dictionary:
            return None

        cluster_name = self.user_dictionary[user_id].cluster

        # We remove the given user from the cluster in order to avoid bias
        cluster_users = list(self.user_cluster_dictionary[cluster_name])
        cluster_users.remove(user_id)

        similarities_sum = 0.
        similarities_ratings_sum = 0.
        num_users = 0
        for cluster_user in cluster_users:
            users_similarity = self.user_similarity_matrix[cluster_user][user_id]

            if item_id in self.user_dictionary[cluster_user].item_ratings and users_similarity is not None:
                cluster_user_item_rating = self.user_dictionary[cluster_user].item_ratings[item_id]
                similarities_sum += users_similarity
                similarities_ratings_sum += users_similarity * cluster_user_item_rating
                num_users += 1

        if num_users == 0:
            return None

        predicted_rating = similarities_ratings_sum / similarities_sum

        return predicted_rating

    @property
    def similarity_metric(self):
        return self._similarity_metric

    @similarity_metric.setter
    def similarity_metric(self, similarity_metric):
        self._similarity_metric = similarity_metric
