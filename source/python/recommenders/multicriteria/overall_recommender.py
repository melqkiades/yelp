from recommenders.multicriteria.multicriteria_base_recommender import \
    MultiCriteriaBaseRecommender

__author__ = 'fpena'


class OverallRecommender(MultiCriteriaBaseRecommender):

    def __init__(self, significant_criteria_ranges=None):
        super(OverallRecommender, self).__init__(
            'OverallRecommender',
            similarity_metric=None,
            significant_criteria_ranges=significant_criteria_ranges)

    def predict_rating(self, user_id, item_id):

        if user_id not in self.user_dictionary:
            return None

        cluster_name = self.user_dictionary[user_id].cluster

        # We remove the given user from the cluster in order to avoid bias
        cluster_users = list(self.user_cluster_dictionary[cluster_name])
        cluster_users.remove(user_id)
        similarities_ratings_sum = 0.
        num_users = 0

        for cluster_user in cluster_users:
            if item_id in self.user_dictionary[cluster_user].item_ratings:
                cluster_user_item_rating = self.user_dictionary[cluster_user].item_ratings[item_id]
                similarities_ratings_sum += cluster_user_item_rating
                num_users += 1

        if num_users == 0:
            return None

        predicted_rating = similarities_ratings_sum / num_users

        return predicted_rating