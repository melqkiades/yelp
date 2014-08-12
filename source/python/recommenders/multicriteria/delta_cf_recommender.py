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

        neighbourhood = self.get_neighbourhood(user_id)

        weighted_sum = 0.
        z_denominator = 0.
        num_users = 0

        for neighbour in neighbourhood:

            similarity = self.user_similarity_matrix[neighbour][user_id]

            if item_id in self.user_dictionary[neighbour].item_ratings and similarity is not None:
                neighbour_item_rating = \
                    self.user_dictionary[neighbour].item_ratings[item_id]
                neighbour_average_rating = \
                    self.user_dictionary[neighbour].average_overall_rating

                weighted_sum += similarity * (
                    neighbour_item_rating - neighbour_average_rating)
                z_denominator += abs(similarity)
                num_users += 1

            if num_users == self._num_neighbors:
                break

        if z_denominator == 0:
            return None

        user_average_rating = \
            self.user_dictionary[user_id].average_overall_rating
        predicted_rating = user_average_rating + weighted_sum / z_denominator

        return predicted_rating
