import math
import itertools
from random import shuffle
import scipy
from utils import dictionary_utils
from etl import ETLUtils
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity import recommender_evaluator

__author__ = 'fpena'


class BasicKNN:

    def __init__(self, num_neighbors):
        self.reviews = None
        self.num_neighbors = num_neighbors
        self.ratings_matrix = None
        self.similarity_matrix = None
        self.user_dictionary = None
        self.user_ids = None

    def load(self, reviews):
        self.reviews = reviews
        self.ratings_matrix = create_ratings_matrix(reviews)
        self.user_dictionary = extractor.initialize_users(self.reviews, False)
        self.user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        self.similarity_matrix = self.create_similarity_matrix()

    def get_rating(self, user, item):
        return self.ratings_matrix[user][item]

    def calculate_pearson_similarity(self, user1, user2):

        common_items = self.get_common_rated_items(user1, user2)

        if not common_items:
            return 0

        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for item in common_items:
            user1_rating = self.get_rating(user1, item)
            user2_rating = self.get_rating(user2, item)
            user1_average = self.user_dictionary[user1].average_overall_rating
            user2_average = self.user_dictionary[user2].average_overall_rating

            # print('user average', user1_average)

            numerator +=\
                (user1_rating - user1_average) * (user2_rating - user2_average)
            denominator1 += (user1_rating - user1_average) ** 2
            denominator2 += (user2_rating - user2_average) ** 2

        denominator = math.sqrt(denominator1 * denominator2)

        if denominator == 0:
            return 0

        return numerator / denominator

    def calculate_pearson_similarity2(self, user1, user2):

        common_items = self.get_common_rated_items(user1, user2)

        if not common_items:
            return 0

        user1_ratings = []
        user2_ratings = []

        for item in common_items:
            user1_ratings.append(self.get_rating(user1, item))
            user2_ratings.append(self.get_rating(user2, item))

        similarity = scipy.stats.pearsonr(user1_ratings, user2_ratings)[0]
        if math.isnan(similarity):
            return 0
        # if similarity < 0:
        #     return -similarity
        return similarity

    def calculate_cosine_similarity(self, user1, user2):

        common_items = self.get_common_rated_items(user1, user2)

        if not common_items:
            return None

        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for item in common_items:
            user1_rating = self.get_rating(user1, item)
            user2_rating = self.get_rating(user2, item)

            numerator += user1_rating * user2_rating
            denominator1 += user1_rating ** 2
            denominator2 += user2_rating ** 2

        denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

        # if denominator == 0:
        #     pass

        return numerator / denominator

    def calculate_similarity(self, user1, user2):
        return self.calculate_pearson_similarity(user1, user2)
        # return self.calculate_pearson_similarity2(user1, user2)
        # return self.calculate_cosine_similarity(user1, user2)

    def create_similarity_matrix(self):

        similarity_matrix = {}

        for user in self.user_ids:
            similarity_matrix[user] = {}

        for user1, user2 in itertools.combinations(self.user_ids, 2):
            similarity = self.calculate_similarity(user1, user2)
            similarity_matrix[user1][user2] = similarity
            similarity_matrix[user2][user1] = similarity
            # print('similarity', similarity)

        return similarity_matrix

    def get_common_rated_items(self, user1, user2):
        """
        Obtains the items that user1 and user2 have rated in common

        :param user1:
        :param user2:
        """
        items_user1 = self.user_dictionary[user1].item_ratings.keys()
        items_user2 = self.user_dictionary[user2].item_ratings.keys()

        return list(set(items_user1).intersection(items_user2))

    def get_user_neighbours(self, user, item):

        sim_users_matrix = self.similarity_matrix[user].copy()
        sim_users_matrix.pop(user, None)

        # We remove the users who have not rated the given item
        sim_users_matrix = {
            k: v for k, v in sim_users_matrix.items()
            if item in self.ratings_matrix[k]}

        # We remove neighbours that don't have a similarity with user
        sim_users_matrix = {
            k: v for k, v in sim_users_matrix.items()
            if v}

        # print(sim_users_matrix)

        # Sort the users by similarity
        neighbourhood = dictionary_utils.sort_dictionary_keys(
            sim_users_matrix)  # [:self.num_neighbors]

        if user in neighbourhood:
            print('Help!!!')

        # print('neighbourhood size:', len(neighbourhood))

        return neighbourhood

    def predict_rating(self, user, item):

        if user not in self.user_ids:
            return None

        ratings_sum = 0
        similarities_sum = 0
        num_users = 0
        neighbourhood = self.get_user_neighbours(user, item)
        # print(neighbourhood)

        if not neighbourhood:
            return None

        # print(neighbourhood)

        for neighbour in neighbourhood:

            similarity = self.calculate_similarity(user, neighbour)
            print('similarity', similarity)

            if item in self.user_dictionary[neighbour].item_ratings and similarity is not None:

                neighbor_rating = self.get_rating(neighbour, item)
                neighbor_average = \
                    self.user_dictionary[neighbour].average_overall_rating
                ratings_sum += similarity * (neighbor_rating - neighbor_average)
                similarities_sum += abs(similarity)
                num_users += 1

            if num_users == self.num_neighbors:
                break

        if similarities_sum == 0:
            return None

        k = 1 / similarities_sum
        user_average = self.user_dictionary[user].average_overall_rating

        predicted_rating = user_average + k * ratings_sum

        return predicted_rating


def load_data(json_file):
    records = ETLUtils.load_json_file(json_file)
    fields = ['user_id', 'business_id', 'stars']
    records = ETLUtils.select_fields(fields, records)

    # We rename the 'stars' field to 'overall_rating' to take advantage of the
    # function extractor.get_user_average_overall_rating
    for record in records:
        record['overall_rating'] = record.pop('stars')
        record['offering_id'] = record.pop('business_id')

    return records


def create_ratings_matrix(records):
    """
    Creates a dictionary of dictionaries with all the ratings available in the
    records. A rating can then be accessed by using ratings_matrix[user][item].
    The goal of this method is to generate a data structure in which the ratings
    can be queried in constant time.

    Beware that this method assumes that there is only one rating per user-item
    pair, in case there is more than one, only the last rating found in the
    records list will be present on the matrix, the rest will be ignored

    :type records: list[dict]
    :param records: a list of  dictionaries, in which each record contains three
    fields: 'user_id', 'business_id' and 'rating'
    :rtype: dict[dict]
    :return: a dictionary of dictionaries with all the ratings
    """
    ratings_matrix = {}

    for record in records:
        user = record['user_id']
        item = record['offering_id']
        rating = record['overall_rating']

        if user not in ratings_matrix:
            ratings_matrix[user] = {}
        ratings_matrix[user][item] = rating

    return ratings_matrix

def main():

    reviews_file =\
        "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review_hotels.json"
    my_records = load_data(reviews_file)
    my_ratings_matrix = create_ratings_matrix(my_records)

    my_records = extractor.remove_users_with_low_reviews(my_records, 1)

    print(len(my_records))
    # print(len(my_ratings_matrix))

    # basic_knn = BasicKNN(1)
    # basic_knn.load(my_records)

    # for record in my_records:
    #     print(basic_knn.predict_rating(record['user_id'], record['offering_id']))

    # print(basic_knn.predict_rating('qLCpuCWCyPb4G2vN-WZz-Q', '8ZwO9VuLDWJOXmtAdc7LXQ')) # 4
    # print(basic_knn.predict_rating('rVlgz-MGYRPa8UzTYO0RGQ', 'c0iszTWZwYtO3TgBx0Z0fQ')) # 2
    # print(basic_knn.predict_rating('4o7r-QSYhOkxpxRMqpXcCg', 'EcHuaHD9IcoPEWNsU8vDTw')) # 4
    # print(basic_knn.predict_rating('msgAEWFbD4df0EvyOR3TnQ', 'EcHuaHD9IcoPEWNsU8vDTw')) # 5

    shuffle(my_records)

    # Split 80-20 and see the results

    num_records = len(my_records)
    num_unknown_records = 0
    training_size = int(num_records*0.8)
    my_train_data = my_records[:training_size]
    my_test_data = my_records[training_size:]

    basic_knn = BasicKNN(None)
    basic_knn.load(my_train_data)
    # basic_knn.load(my_records)
    recommender_evaluator.perform_cross_validation(my_records, basic_knn, 3)

    # num_items = 0.0
    # for my_user in basic_knn.user_dictionary.values():
    #     print(len(my_user.item_ratings))
    #     num_items += len(my_user.item_ratings)
    #
    # print('average_ratings', (num_items/len(basic_knn.user_dictionary)))


    # for record in my_test_data:
    #     my_prediction = basic_knn.predict_rating(record['user_id'], record['offering_id'])
    #     print(my_prediction)

# main()
