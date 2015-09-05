import math
from recommenders.context.similarity.base_similarity_calculator import \
    BaseSimilarityCalculator
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class PearsonSimilarityCalculator(BaseSimilarityCalculator):
    def __init__(self):
        super(PearsonSimilarityCalculator, self).__init__()

    def calculate_user_similarity(self, user1, user2, threshold):
        common_items =\
            extractor.get_common_items(self.user_dictionary, user1, user2)

        if not common_items:
            return None

        numerator = 0
        denominator1 = 0
        denominator2 = 0

        user1_average = self.user_dictionary[user1].average_overall_rating
        user2_average = self.user_dictionary[user2].average_overall_rating

        for item in common_items:
            user1_rating = self.user_dictionary[user1].item_ratings[item]
            user2_rating = self.user_dictionary[user2].item_ratings[item]

            # print('user average', user1_average)

            numerator +=\
                (user1_rating - user1_average) * (user2_rating - user2_average)
            denominator1 += (user1_rating - user1_average) ** 2
            denominator2 += (user2_rating - user2_average) ** 2

        denominator = math.sqrt(denominator1 * denominator2)

        if denominator == 0:
            return 0

        return numerator / denominator
