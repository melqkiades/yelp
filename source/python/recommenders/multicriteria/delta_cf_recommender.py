from recommenders.multicriteria.multicriteria_base_recommender import \
    MultiCriteriaBaseRecommender

__author__ = 'fpena'


class DeltaCFRecommender(MultiCriteriaBaseRecommender):

    def __init__(
            self, similarity_metric='euclidean', significant_criteria_ranges=None):
        super(DeltaCFRecommender, self).__init__(
            'DeltaCFRecommender',
            similarity_metric=similarity_metric,
            significant_criteria_ranges=significant_criteria_ranges)

    def predict_rating(self, user_id, item_id):
        """
        Predicts the rating the user will give to the hotel

        :param user_id: the ID of the user
        :param item_id: the ID of the hotel
        :return: a float between 1 and 5 with the predicted rating
        """
        if user_id not in self.user_ids:
            return None

        other_users = self.get_most_similar_users(user_id)

        weighted_sum = 0.
        z_denominator = 0.

        for other_user in other_users:

            similarity = self.user_similarity_matrix[other_user][user_id]

            if item_id in self.user_dictionary[other_user].item_ratings and similarity is not None:
                other_user_item_rating = \
                    self.user_dictionary[other_user].item_ratings[item_id]
                other_user_average_rating = \
                    self.user_dictionary[other_user].average_overall_rating

                weighted_sum += similarity * (
                    other_user_item_rating - other_user_average_rating)
                z_denominator += abs(similarity)

        if z_denominator == 0:
            return None

        user_average_rating = \
            self.user_dictionary[user_id].average_overall_rating
        predicted_rating = user_average_rating + weighted_sum / z_denominator

        return predicted_rating
