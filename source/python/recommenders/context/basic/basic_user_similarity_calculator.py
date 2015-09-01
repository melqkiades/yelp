import math
import itertools
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class BasicUserSimilarityCalculator:

    def __init__(self):
        self.user_ids = None
        self.user_dictionary = None
        self.context_rich_topics = None

    def load(self, user_ids, user_dictionary, context_rich_topics):
        self.user_ids = user_ids
        self.user_dictionary = user_dictionary
        self.context_rich_topics = context_rich_topics

    def calculate_pearson_similarity(self, user1, user2):

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

    def calculate_cosine_similarity(self, user1, user2):

        common_items =\
            extractor.get_common_items(self.user_dictionary, user1, user2)

        if not common_items:
            return None

        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for item in common_items:
            user1_rating = self.user_dictionary[user1].item_ratings[item]
            user2_rating = self.user_dictionary[user2].item_ratings[item]

            numerator += user1_rating * user2_rating
            denominator1 += user1_rating ** 2
            denominator2 += user2_rating ** 2

        denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

        # if denominator == 0:
        #     pass

        return numerator / denominator

    def calculate_user_similarity(self, user1, user2, threshold):
        # return self.calculate_pearson_similarity(user1, user2)
        # return self.calculate_pearson_similarity2(user1, user2)
        return self.calculate_cosine_similarity(user1, user2)

    # def get_common_rated_items(self, user1, user2):
    #     """
    #     Obtains the items that user1 and user2 have rated in common
    #
    #     :param user1:
    #     :param user2:
    #     """
    #     items_user1 = self.user_dictionary[user1].item_ratings.keys()
    #     items_user2 = self.user_dictionary[user2].item_ratings.keys()
    #
    #     return list(set(items_user1).intersection(items_user2))

    def create_similarity_matrix(self, threshold):

        similarity_matrix = {}

        for user in self.user_ids:
            similarity_matrix[user] = {}

        for user_id1, user_id2 in itertools.combinations(self.user_ids, 2):
            similarity =\
                self.calculate_user_similarity(user_id1, user_id2, None)
            similarity_matrix[user_id1][user_id2] = similarity
            similarity_matrix[user_id2][user_id1] = similarity

            # print('similarity', user_id1, user_id2, similarity)

        return similarity_matrix
