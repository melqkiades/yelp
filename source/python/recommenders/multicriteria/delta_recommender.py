from recommenders.multicriteria.multicriteria_base_recommender import \
    MultiCriteriaBaseRecommender

__author__ = 'fpena'


class DeltaRecommender(MultiCriteriaBaseRecommender):

    def __init__(self, significant_criteria_ranges=None):
        super(DeltaRecommender, self).__init__(
            'DeltaRecommender',
            similarity_metric=None,
            significant_criteria_ranges=significant_criteria_ranges)

    def predict_rating(self, user_id, item_id):
        """
        Predicts the rating the user will give to the hotel

        :param user_id: the ID of the user
        :param item_id: the ID of the hotel
        :return: a float between 1 and 5 with the predicted rating
        """
        if user_id not in self.user_dictionary:
            return None

        cluster_name = self.user_dictionary[user_id].cluster

        # We remove the given user from the cluster in order to avoid bias
        cluster_users = list(self.user_cluster_dictionary[cluster_name])
        cluster_users.remove(user_id)
        similarities_ratings_sum = 0.
        num_users = 0

        for cluster_user in cluster_users:
            cluster_user_overall_rating = self.user_dictionary[cluster_user].average_overall_rating

            if item_id in self.user_dictionary[cluster_user].item_ratings:
                cluster_user_item_rating = self.user_dictionary[cluster_user].item_ratings[item_id]
                similarities_ratings_sum += (cluster_user_item_rating - cluster_user_overall_rating)
                num_users += 1

        if num_users == 0:
            return None

        user_average_rating = self.user_dictionary[user_id].average_overall_rating
        predicted_rating = \
            user_average_rating + similarities_ratings_sum / num_users

        return predicted_rating