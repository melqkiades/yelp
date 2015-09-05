import math
from recommenders.context.similarity.base_similarity_calculator import \
    BaseSimilarityCalculator
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class CosineSimilarityCalculator(BaseSimilarityCalculator):
    def __init__(self):
        super(CosineSimilarityCalculator, self).__init__()

    def calculate_user_similarity(self, user_id1, user_id2, threshold):
        common_items =\
            extractor.get_common_items(self.user_dictionary, user_id1, user_id2)

        if not common_items:
            return None

        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for item in common_items:
            user1_rating = self.user_dictionary[user_id1].item_ratings[item]
            user2_rating = self.user_dictionary[user_id2].item_ratings[item]

            numerator += user1_rating * user2_rating
            denominator1 += user1_rating ** 2
            denominator2 += user2_rating ** 2

        denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

        return numerator / denominator
