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

        if user_id not in self.user_ids:
            return None

        other_users = self.get_most_similar_users(user_id)

        weighted_sum = 0.
        z_denominator = 0.

        for other_user in other_users:
            similarity = self.user_similarity_matrix[other_user][user_id]

            if item_id in self.user_dictionary[other_user].item_ratings and similarity is not None:
                other_user_item_rating =\
                    self.user_dictionary[other_user].item_ratings[item_id]
                weighted_sum += similarity * other_user_item_rating
                z_denominator += abs(similarity)

        if z_denominator == 0:
            return None

        predicted_rating = weighted_sum / z_denominator

        return predicted_rating
