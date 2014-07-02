import math
import numpy
import time
from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'




def clu_all_criteria(reviews, user_id, user_cluster_dictionary):
    weights = extractor.get_criteria_weights(reviews, user_id)
    significant_criteria, cluster_name = extractor.get_significant_criteria(weights)
    cluster_users = user_cluster_dictionary[cluster_name].remove(user_id)

    index = 0
    overall_rating = 0

    for weight in weights:
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])

        ratings_sum = 0
        for review in user_reviews:
            ratings_sum += review['rating_list'][index]

        average_rating = ratings_sum / len(user_reviews)
        overall_rating = weight * average_rating
        index += 1


def clu_overall(reviews, user_id, user_cluster_dictionary, hotel_id):

    actual_overall_rating = get_user_item_overall_rating(reviews, user_id, hotel_id)
    weights = extractor.get_criteria_weights(reviews, user_id)
    significant_criteria, cluster_name = extractor.get_significant_criteria(weights)

    # print(cluster_name, significant_criteria)
    # print('User ID: ' + user_id + '\tHotel ID: %i' % hotel_id)
    # print(user_cluster_dictionary[cluster_name])

    # We remove the given user from the cluster in order to avoid bias
    cluster_users = list(user_cluster_dictionary[cluster_name])
    cluster_users.remove(user_id)

    filtered_reviews = ETLUtils.filter_records(reviews, 'offering_id',
                                               [hotel_id])
    filtered_reviews = ETLUtils.filter_out_records(filtered_reviews, 'user_id',
                                                   [user_id])
    # print('Filtered reviews: ' + str(filtered_reviews))
    # print('Cluster ' + cluster_name + ' users: ' + str(cluster_users))
    ratings_sum = 0
    ratings_count = 0
    for user in cluster_users:

        user_reviews = ETLUtils.filter_records(filtered_reviews, 'user_id',
                                               [user])
        # print('User reviews: ' + str(user_reviews))

        for review in user_reviews:
            ratings_sum += review['overall_rating']
            ratings_count += 1

    predicted_rating = None
    error = None
    if ratings_count > 0:
        predicted_rating = float(ratings_sum) / float(ratings_count)
        error = abs(predicted_rating - actual_overall_rating)
        # print('Ratings sum: %d\tRatings Count: %d' % (ratings_sum, ratings_count))
        # print('Predicted rating: %f ' % predicted_rating)
        # print('Error: %f' % error)

    # if ratings_sum != 0:
    #     print('Predicted rating: %d' % predicted_rating)
    #     print('Ratings sum: %d' % ratings_sum)
    #     print(cluster_name, significant_criteria)
    #     print('User ID: ' + user_id + '\tHotel ID: %i' % hotel_id)
    #     print(user_cluster_dictionary[cluster_name])
    #     print('Cluster users: ' + str(cluster_users))

    return predicted_rating, error


def clu_overall_list(reviews, user_cluster_dictionary):
    average_ratings = []
    errors = []

    for review in reviews:
        average_rating, error = clu_overall(
            reviews, review['user_id'], user_cluster_dictionary,
            review['offering_id'])
        average_ratings.append(average_rating)
        errors.append(error)

    return average_ratings, errors


def build_user_clusters(reviews):
    """
    Builds a series of clusters for users according to their significant
    criteria. Users that have exactly the same significant criteria will belong
    to the same cluster.

    :param reviews: the list of reviews
    :return: a dictionary where all the keys are the cluster names and the
    values for those keys are list of users that belong to that cluster
    """

    min_reviews = 10
    user_list = extractor.get_user_list(reviews, min_reviews)
    user_cluster_dictionary = {}

    for user in user_list:
        weights = extractor.get_criteria_weights(reviews, user)
        significant_criteria, cluster_name = extractor.get_significant_criteria(weights)
        # print(significant_criteria)
        # print(cluster_name)

        if cluster_name in user_cluster_dictionary:
            user_cluster_dictionary[cluster_name].append(user)
        else:
            user_cluster_dictionary[cluster_name] = [user]
            # print(user, cluster_name)

    # for key in user_cluster_dictionary.keys():
    #     print(key, len(user_cluster_dictionary[key]))

    return user_cluster_dictionary


def calculate_mean_average_error(errors):
    """
    Calculates the mean average error for the predicted rating

    :param reviews: the list of all reviews
    :param user_cluster_dictionary: a dictionary where all the keys are the
    cluster names and the values for those keys are list of users that belong to
    that cluster
    :return: the mean average error after predicting all the overall ratings
    """
    num_ratings = 0.
    total_error = 0.

    for error in errors:
        if error is not None:
            total_error += error
            num_ratings += 1

    mean_absolute_error = total_error / num_ratings
    return mean_absolute_error


def calculate_root_mean_square_error(errors):
    """
    Calculates the mean average error for the predicted rating

    :param reviews: the list of all reviews
    :param user_cluster_dictionary: a dictionary where all the keys are the
    cluster names and the values for those keys are list of users that belong to
    that cluster
    :return: the mean average error after predicting all the overall ratings
    """
    num_ratings = 0.
    total_error = 0.

    for error in errors:
        if error is not None:
            total_error += error ** 2
            num_ratings += 1

    root_mean_square_error = (total_error / num_ratings) ** 0.5
    return root_mean_square_error


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


# def calculate_users_similarity(reviews, user_id1, user_id2):
#     """
#     Calculates the similarity between two users based on how similar are their
#     ratings in the reviews
#
#     :param reviews: a list of all reviews in the system
#     :param user_id1: the ID of user 1
#     :param user_id2: the ID of user 2
#     :return: a float with the similarity between the two users. Since this
#     function is based on euclidean distance to calculate the similarity, a
#     similarity of 0 indicates that the users share exactly the same tastes
#     """
#     user_weights1 = get_criteria_weights(reviews, user_id1)
#     user_weights2 = get_criteria_weights(reviews, user_id2)
#     # user_weights1 = [1]
#     # user_weights2 = [1]
#
#     return calculate_euclidean_distance(user_weights1, user_weights2)


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

    user_reviews_dictionary = {}

    for user in users:
        user_reviews_dictionary[user] =\
            ETLUtils.filter_records(reviews, 'user_id', user)

    return user_reviews_dictionary



def clu_cf_euc(reviews, user_id, user_cluster_dictionary, hotel_id):

    average_overall_rating = extractor.get_user_average_overall_rating(reviews,
                                                                       user_id)
    # average_overall_rating = 5
    weights = extractor.get_criteria_weights(reviews, user_id)
    significant_criteria, cluster_name = extractor.get_significant_criteria(weights)
    # We remove the given user from the cluster in order to avoid bias
    cluster_users = list(user_cluster_dictionary[cluster_name])
    cluster_users.remove(user_id)

    similarities_sum = 1.
    similarities_ratings_sum = 1.
    for cluster_user in cluster_users:
        # users_similarity = calculate_users_similarity(reviews, cluster_user,
        #                                                user_id)
        users_similarity = 0
        similarities_sum += users_similarity
        # user_item_rating = get_user_item_overall_rating(reviews, user_id,
        #                                                 hotel_id)
        user_item_rating = 4
        similarities_ratings_sum += users_similarity * user_item_rating
    predicted_rating = None
    error = None

    if cluster_users:
        predicted_rating =\
            average_overall_rating * similarities_ratings_sum / similarities_sum
        error = abs(predicted_rating - average_overall_rating)

    return predicted_rating, error


def clu_cf_euc_list(reviews, user_cluster_dictionary):
    predicted_ratings = []
    errors = []

    index = 0
    print('Total reviews: %i' % len(reviews))

    for review in reviews:
        print('Index: %i' % index)
        index += 1

        predicted_rating, error = clu_cf_euc(
            reviews, review['user_id'], user_cluster_dictionary,
            review['offering_id'])
        # predicted_rating = 4
        # error = 0
        predicted_ratings.append(predicted_rating)
        errors.append(error)

    return predicted_ratings, errors


def main():
    # reviews = extractor.pre_process_reviews()
    # user_cluster_dictionary = build_user_clusters(reviews)
    # _, errors = clu_cf_euc_list(reviews, user_cluster_dictionary)
    # mean_absolute_error = calculate_mean_average_error(errors)
    # print('Mean Absolute error: %f' % mean_absolute_error)
    # root_mean_square_error = calculate_root_mean_square_error(errors)
    # print('Root mean square error: %f' % root_mean_square_error)
    pass


# start_time = time.time()
# main()
# end_time = time.time() - start_time
# print("--- %s seconds ---" % end_time)