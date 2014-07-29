from tripadvisor.fourcity.base_recommender import BaseRecommender

__author__ = 'fpena'


class AdjustedWeightedSumRecommender(BaseRecommender):
    def __init__(self, similarity_metric='cosine'):
        super(AdjustedWeightedSumRecommender, self).__init__('SingleCF',
                                                             'cosine')
        self.similarity_metric = similarity_metric

    def predict_rating(self, user_id, item_id):

        other_users = list(self.user_ids)

        if user_id not in other_users:
            return None

        other_users.remove(user_id)
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

    @property
    def name(self):
        return 'Single_CF'