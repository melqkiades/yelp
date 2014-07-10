import math
from scipy import spatial
from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'




def clu_overall(reviews, user_id, user_cluster_dictionary, hotel_id):

    actual_user_item_rating =\
        get_user_item_overall_rating(reviews, user_id, hotel_id)
    weights =\
        extractor.get_criteria_weights(reviews, user_id)
    significant_criteria, cluster_name =\
        extractor.get_significant_criteria(weights)

    # We remove the given user from the cluster in order to avoid bias
    cluster_users = list(user_cluster_dictionary[cluster_name])
    cluster_users.remove(user_id)

    filtered_reviews = ETLUtils.filter_records(reviews, 'offering_id',
                                               [hotel_id])
    filtered_reviews = ETLUtils.filter_out_records(filtered_reviews, 'user_id',
                                                   [user_id])
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
        # predicted_rating = 3.9
        predicted_rating = float(ratings_sum) / float(ratings_count)
        error = abs(predicted_rating - actual_user_item_rating)

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


def main():
    # reviews = extractor.pre_process_reviews()
    reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
    user_cluster_dictionary = build_user_clusters(reviews)
    _, errors = clu_overall_list(reviews, user_cluster_dictionary)
    # _, errors = clu_cf_euc_list(reviews, user_cluster_dictionary)
    # mean_absolute_error = calculate_mean_average_error(errors)
    # print('Mean Absolute error: %f' % mean_absolute_error)
    # root_mean_square_error = calculate_root_mean_square_error(errors)
    # print('Root mean square error: %f' % root_mean_square_error)
    pass


# start_time = time.time()
# perform_cross_validation()
# # main()
# end_time = time.time() - start_time
# print("--- %s seconds ---" % end_time)

# x1 = [1, 2, 4, 4, 5]
# x2 = [1, 2, 3, 4, 5]
#
# print(calculate_euclidean_distance(x1, x2))
# print(spatial.distance.cosine(x1, x2))