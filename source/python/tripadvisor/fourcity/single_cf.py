from etl import similarity
from tripadvisor.fourcity.reviews_holder import ReviewsHolder

__author__ = 'fpena'


class SingleCF(object):

    def __init__(self, reviews):
        self.reviews = reviews
        self.reviews_holder = ReviewsHolder(self.reviews)
        self.user_ids = self.reviews_holder.user_ids

    def predict_rating(self, user_id, item_id):
        # return self.calculate_weighted_sum(user_id, item_id)
        return self.calculate_adjusted_weighted_sum(user_id, item_id)

    def calculate_weighted_sum(self, user_id, item_id):

        other_users = list(self.user_ids)

        if user_id not in other_users:
            return None

        other_users.remove(user_id)
        weighted_sum = 0.
        z_denominator = 0.

        for other_user in other_users:
            other_user_item_rating = self.reviews_holder.get_rating(other_user, item_id)
            similarity = self.calculate_similarity(user_id, other_user)

            if other_user_item_rating is None or not similarity:
                continue

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
            other_user_item_rating = self.reviews_holder.get_rating(other_user, item_id)
            other_user_average_rating = self.reviews_holder.get_user_average_rating(other_user)
            similarity = self.calculate_similarity(user_id, other_user)

            if other_user_item_rating is None or not similarity:
                continue

            weighted_sum += similarity * (other_user_item_rating - other_user_average_rating)
            z_denominator += abs(similarity)

        if z_denominator == 0:
            return None

        user_average_rating = self.reviews_holder.get_user_average_rating(user_id)
        predicted_rating = user_average_rating + weighted_sum / z_denominator

        print('Predicted rating', predicted_rating)

        return predicted_rating

    def calculate_similarity(self, user1, user2):
        common_items = self.reviews_holder.get_common_items(user1, user2)

        if not common_items:
            return None

        user1_ratings = self.reviews_holder.extract_user_ratings(user1, common_items)
        user2_ratings = self.reviews_holder.extract_user_ratings(user2, common_items)

        similarity_value = similarity.cosine(user1_ratings, user2_ratings)

        return similarity_value


    @property
    def name(self):
        return 'Single_CF'