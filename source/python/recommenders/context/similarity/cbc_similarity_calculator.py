import math
from recommenders.context.similarity.base_similarity_calculator import \
    BaseSimilarityCalculator
from topicmodeling.context import context_utils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class CBCSimilarityCalculator(BaseSimilarityCalculator):
    def __init__(self):
        super(CBCSimilarityCalculator, self).__init__()

    def calculate_user_similarity(self, user1, user2, threshold):

        common_items = extractor.get_common_items(
            self.user_dictionary, user1, user2)

        if not common_items:
            return None

        filtered_items = {}

        for item in common_items:
            context1 = self.user_dictionary[user1].item_contexts[item]
            context2 = self.user_dictionary[user2].item_contexts[item]
            context_similarity = context_utils.get_context_similarity(
                context1, context2, self.context_rich_topics)
            if context_similarity > threshold:
                filtered_items[item] = context_similarity

        numerator = 0
        denominator1 = 0
        denominator2 = 0
        denominator3 = 0
        user1_average = self.user_dictionary[user1].average_overall_rating
        user2_average = self.user_dictionary[user2].average_overall_rating

        for item in filtered_items.keys():
            context_similarity = filtered_items[item]
            user1_rating = self.user_dictionary[user1].item_ratings[item]
            user2_rating = self.user_dictionary[user2].item_ratings[item]

            # numerator +=\
            #     (user1_rating - user1_average) *\
            #     (user2_rating - user2_average) *\
            #     context_similarity
            # denominator1 += (user1_rating - user1_average) ** 2
            # denominator2 += (user2_rating - user2_average) ** 2
            numerator += user1_rating * user2_rating * context_similarity
            denominator1 += user1_rating ** 2
            denominator2 += user2_rating ** 2
            denominator3 += context_similarity ** 2

        denominator = math.sqrt(denominator1) * math.sqrt(denominator2) * math.sqrt(denominator3)

        # return self.calculate_cosine_similarity(user1, user2)

        if denominator == 0:
            return None
        #
        return numerator / denominator
        # return self.calculate_pearson_similarity(user1, user2)
        # return self.calculate_cosine_similarity(user1, user2)