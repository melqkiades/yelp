from tripadvisor.fourcity import fourcity_clusterer
from tripadvisor.fourcity.base_recommender import BaseRecommender

__author__ = 'fpena'


class SingleCF(BaseRecommender):

    def __init__(self, similarity_metric='cosine'):
        super(SingleCF, self).__init__('SingleCF', 'cosine')
        self.similarity_metric = similarity_metric

    def predict_rating(self, user_id, item_id):
        return self.calculate_weighted_sum(user_id, item_id)
        # return self.calculate_adjusted_weighted_sum(user_id, item_id)

    def calculate_weighted_sum(self, user_id, item_id):

        other_users = list(self.user_ids)

        if user_id not in other_users:
            return None

        other_users.remove(user_id)
        weighted_sum = 0.
        z_denominator = 0.

        for other_user in other_users:
            # other_user_item_rating = self.reviews_holder.get_rating(other_user, item_id)
            # similarity = self.calculate_similarity(user_id, other_user)
            similarity = self.user_similarity_matrix[other_user][user_id]

            if item_id in self.user_dictionary[other_user].item_ratings and similarity is not None:
                other_user_item_rating = self.user_dictionary[other_user].item_ratings[item_id]
                weighted_sum += similarity * other_user_item_rating
                z_denominator += abs(similarity)

        if z_denominator == 0:
            return None

        predicted_rating = weighted_sum / z_denominator

        return predicted_rating

    def calculate_adjusted_weighted_sum(self, user_id, item_id):

        other_users = list(self.user_ids)

        if user_id not in other_users:
            return None

        other_users.remove(user_id)
        weighted_sum = 0.
        z_denominator = 0.

        for other_user in other_users:
            # other_user_item_rating = self.reviews_holder.get_rating(other_user, item_id)
            # other_user_average_rating = self.reviews_holder.get_user_average_rating(other_user)
            other_user_item_rating = self.user_dictionary[other_user].item_ratings[item_id]
            other_user_average_rating = self.user_dictionary[other_user].average_overall_rating

            # similarity = self.calculate_similarity(user_id, other_user)
            similarity = self.user_similarity_matrix[other_user][user_id]

            if other_user_item_rating is None or not similarity:
                continue

            weighted_sum += similarity * (other_user_item_rating - other_user_average_rating)
            z_denominator += abs(similarity)

        if z_denominator == 0:
            return None

        # user_average_rating = self.reviews_holder.get_user_average_rating(user_id)
        user_average_rating = self.user_dictionary[user_id].average_overall_rating
        predicted_rating = user_average_rating + weighted_sum / z_denominator

        # print('Predicted rating', predicted_rating)

        return predicted_rating

    @property
    def name(self):
        return 'Single_CF'