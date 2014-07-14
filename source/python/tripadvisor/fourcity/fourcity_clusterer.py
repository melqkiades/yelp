import math
from scipy import spatial
from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


def build_user_clusters(reviews):
    """
    Builds a series of clusters for users according to their significant
    criteria. Users that have exactly the same significant criteria will belong
    to the same cluster.

    :param reviews: the list of reviews
    :return: a dictionary where all the keys are the cluster names and the
    values for those keys are list of users that belong to that cluster
    """

    user_list = extractor.get_groupby_list(reviews, 'user_id')
    user_cluster_dictionary = {}

    for user in user_list:
        weights = extractor.get_criteria_weights(reviews, user)
        significant_criteria, cluster_name =\
            extractor.get_significant_criteria(weights)

        if cluster_name in user_cluster_dictionary:
            user_cluster_dictionary[cluster_name].append(user)
        else:
            user_cluster_dictionary[cluster_name] = [user]

    return user_cluster_dictionary


def calculate_euclidean_distance(vector1, vector2):
    """
    Calculates the euclidean distance between two vectors of dimension N

    :param vector1: a list of numeric values of size N
    :param vector2: a list of numeric values of size N
    :return: a float with the euclidean distance between vector1 and vector2
    """
    distance = 0.
    for value1, value2 in zip(vector1, vector2):
        distance += (value2 - value1) ** 2

    return math.sqrt(distance)


def get_user_item_overall_rating(reviews, user_id, item_id):
    filtered_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    filtered_reviews = ETLUtils.filter_records(filtered_reviews, 'offering_id',
                                               [item_id])

    if not filtered_reviews:
        return None

    overall_rating_sum = 0.

    for review in filtered_reviews:
        overall_rating_sum += review['ratings']['overall']

    user_item_overall_rating = overall_rating_sum / len(filtered_reviews)
    return user_item_overall_rating


def build_user_reviews_dictionary(reviews, users):
    """
    Builds a dictionary that contains all the reviews the users have made where
    the key is the user ID and the value is a list of the reviews this user has
    made.

    :param reviews: a list of reviews
    :param users: the list of users to be considered
    :return: a dictionary that contains all the reviews the users have made
    where the key is the user ID and the value is a list of the reviews this
    user has made.
    """
    user_reviews_dictionary = {}

    for user in users:
        user_reviews_dictionary[user] =\
            ETLUtils.filter_records(reviews, 'user_id', user)

    return user_reviews_dictionary


def calculate_users_similarity(user_dictionary, user_id1, user_id2):
    """
    Calculates the similarity between two users based on how similar are their
    ratings in the reviews

    :param user_id1: the ID of user 1
    :param user_id2: the ID of user 2
    :return: a float with the similarity between the two users. Since this
    function is based on euclidean distance to calculate the similarity, a
    similarity of 0 indicates that the users share exactly the same tastes
    """
    user_weights1 = user_dictionary[user_id1].criteria_weights
    user_weights2 = user_dictionary[user_id2].criteria_weights

    return calculate_euclidean_distance(user_weights1, user_weights2)
    # return spatial.distance.cosine(user_weights1, user_weights2)
    # return 0


def build_user_similarities_matrix(user_ids, user_dictionary):
    """
    Builds a matrix that contains the similarity between every pair of users
    in the dataset of this recommender system. This is particularly useful
    to prevent repeating the same calculations in each cycle

    """
    user_similarity_matrix = {}

    for user1 in user_ids:
        user_similarity_matrix[user1] = {}
        for user2 in user_ids:
            user_similarity_matrix[user1][user2] =\
                calculate_users_similarity(user_dictionary, user1, user2)

    return user_similarity_matrix


# x1 = [1, 2, 4, 4, 5]
# x2 = [1, 2, 3, 4, 5]
#
# print(calculate_euclidean_distance(x1, x2))
# print(spatial.distance.cosine(x1, x2))