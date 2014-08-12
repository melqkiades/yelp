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

        neighbourhood = self.get_neighbourhood(user_id)
        similarities_ratings_sum = 0.
        num_users = 0

        for neighbour in neighbourhood:
            if item_id in self.user_dictionary[neighbour].item_ratings:
                neighbour_item_rating = \
                    self.user_dictionary[neighbour].item_ratings[item_id]
                similarities_ratings_sum += neighbour_item_rating
                num_users += 1

            if num_users == self._num_neighbors:
                break

        if num_users == 0:
            return None

        predicted_rating = similarities_ratings_sum / num_users

        return predicted_rating