from recommenders.base_recommender import BaseRecommender

__author__ = 'fpena'


class WeightedSumRecommender(BaseRecommender):

    def __init__(self, similarity_matrix_builder, num_neighbors=None):
        super(WeightedSumRecommender, self).__init__(
            'WeightedSumRecommender', similarity_matrix_builder, num_neighbors)

    def predict_rating(self, user_id, item_id):

        if user_id not in self.user_ids:
            return None

        neighbourhood = self.get_neighbourhood(user_id)

        weighted_sum = 0.
        z_denominator = 0.
        num_users = 0

        for neighbour in neighbourhood:
            similarity = self.user_similarity_matrix[neighbour][user_id]

            if item_id in self.user_dictionary[neighbour].item_ratings and similarity is not None:
                neighbour_item_rating =\
                    self.user_dictionary[neighbour].item_ratings[item_id]
                weighted_sum += similarity * neighbour_item_rating
                z_denominator += abs(similarity)
                num_users += 1

            if num_users == self._num_neighbors:
                break

        if z_denominator == 0:
            return None

        predicted_rating = weighted_sum / z_denominator

        return predicted_rating
