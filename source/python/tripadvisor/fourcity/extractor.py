import itertools
import operator

import numpy
from pandas import DataFrame

from etl import ETLUtils
from tripadvisor.fourcity.user import User


__author__ = 'fpena'


def remove_empty_user_reviews(reviews):
    """
    Returns a copy of the original reviews list without the reviews made by
    users who have an empty ID

    :param reviews: a list of reviews
    :return: a copy of the original reviews list without the reviews made by
    users who have an empty ID
    """
    filtered_reviews = [review for review in reviews if
                        review['user_id'] != '']
    return filtered_reviews


def extract_fields(reviews):
    """
    Modifies the given list of reviews in order to extract the values contained
    in the ratings field to top level fields. For instance, a review which is
    in the form
    {'user_id': 'U1', 'offering_id': :'I1',
    'ratings': {'cleanliness': 4.0, 'location': 5.0}}
    would become:

    {'user_id': 'U1', 'offering_id': :'I1',
    'ratings': {'cleanliness': 4.0, 'location': 5.0},
    'cleanliness_rating': 4.0, 'location_rating': 5.0}

    :param reviews: a list of reviews.
    """

    ratings_criteria = [
        'cleanliness',
        'location',
        'rooms',
        'service',
        # 'sleep_quality',
        'value'
    ]

    for review in reviews:
        review['user_id'] = review['author']['id']
        review['overall_rating'] = review['ratings']['overall']

        ratings = review['ratings']
        multi_ratings = []
        for criterion in ratings_criteria:
            if criterion in ratings:
                # review[criterion + '_rating'] = review['ratings'][criterion]
                multi_ratings.append(review['ratings'][criterion])
            else:
                multi_ratings.append(-1)

        review['multi_ratings'] = multi_ratings


def remove_users_with_low_reviews(reviews, min_reviews):
    """
    Returns a copy of the original reviews list without the reviews made by
    users who have made less than min_reviews reviews

    :param reviews: a list of reviews
    :param min_reviews: the minimum number of reviews a user must have in order
    not to be removed from the reviews list
    :return: a copy of the original reviews list without the reviews made by
    users who have made less than min_reviews reviews
    """
    users = get_user_list(reviews, min_reviews)
    return ETLUtils.filter_records(reviews, 'user_id', users)


def remove_items_with_low_reviews(reviews, min_reviews):
    """
    Returns a copy of the original reviews list without the reviews of hotels
    that just have been reviewed once

    :param reviews: a list of reviews
    :return: a copy of the original reviews list without the reviews of hotels
    that just have been reviewed once
    """
    items = get_item_list(reviews, min_reviews)
    return ETLUtils.filter_records(reviews, 'offering_id', items)


def remove_missing_ratings_reviews(reviews):
    """
    Returns a copy of the original reviews list without the reviews that have
    missing ratings

    :param reviews: a list of reviews
    :return: a copy of the original reviews list without the reviews that have
    missing ratings
    """
    filtered_reviews = [review for review in reviews if
                        verify_rating_criteria(review)]
    return filtered_reviews


def verify_rating_criteria(review):
    """
    Verifies if the given review contains all the ratings criteria required.
    Returns True in case the review contains all the necessary keys. Returns
    False otherwise. For example, if a review contains
    {'ratings': {'cleanliness':3, 'location':4, 'rooms': 5, 'service':3, 'value': 4}, ...}
    It will return True because all the desired criteria is present. But if a
    review contains {'ratings': {'cleanliness':3, 'value': 4}, ...} it will
    return False because it doesn't contain values for 'location', 'rooms' and
    'service'

    :param review: a dictionary with the review information
    :return: True in case all the desired ratings criteria are present in the
    review, False otherwise
    """

    return -1 not in review['multi_ratings']


def clean_reviews(reviews):
    """
    Returns a copy of the original reviews list with only that are useful for
    recommendation purposes

    :param reviews: a list of reviews
    :return: a copy of the original reviews list with only that are useful for
    recommendation purposes
    """
    filtered_reviews = remove_empty_user_reviews(reviews)
    filtered_reviews = remove_missing_ratings_reviews(filtered_reviews)
    print('Finished remove_missing_ratings_reviews')
    filtered_reviews = remove_users_with_low_reviews(filtered_reviews, 10)
    print('Finished remove_users_with_low_reviews')
    filtered_reviews = remove_items_with_low_reviews(filtered_reviews, 20)
    print('Finished remove_single_review_hotels')
    # filtered_reviews = remove_users_with_low_reviews(filtered_reviews, 10)
    # print('Finished remove_users_with_low_reviews')
    print('Number of reviews', len(filtered_reviews))
    return filtered_reviews


def pre_process_reviews():
    """
    Returns a list of preprocessed reviews, where the reviews have been filtered
    to obtain only relevant data, have dropped any fields that are not useful,
    and also have additional fields that are handy to make calculations

    :return: a list of preprocessed reviews
    """
    data_folder = '../../../../../../datasets/TripAdvisor/Four-City/'
    review_file_path = data_folder + 'review.txt'
    # review_file_path = data_folder + 'review-short.json'
    reviews = ETLUtils.load_json_file(review_file_path)

    select_fields = ['ratings', 'author', 'offering_id']
    reviews = ETLUtils.select_fields(select_fields, reviews)
    extract_fields(reviews)
    ETLUtils.drop_fields(['author', 'ratings'], reviews)
    # reviews = load_json_file('/Users/fpena/tmp/filtered_reviews.json')
    # reviews = preflib_extractor.load_csv_file('/Users/fpena/UCC/Thesis/datasets/TripAdvisor/PrefLib/trip/CD-00001-00000001-copy.dat')
    reviews = clean_reviews(reviews)

    return reviews


def create_ratings_matrix(reviews):
    """
    Returns (ratings_matrix, overall_ratings_list), where ratings_matrix is a
    list of lists containing the values for all the rating criteria (except
    the overall rating), so for each review a list of rating values is returned.
    overall_ratings_list is a list containing the overall rating for each of the
    reviews. This function verifies that reviews don't have any missing ratings,
    in case there are ratings missing, those are not included in the returning
    values

    :param reviews: a list of reviews
    :return: (ratings_matrix, overall_ratings_list), where ratings_matrix is a
    list of lists containing the values for all the rating criteria (except
    the overall rating), and overall_ratings_list is a list containing the
    overall rating for each of the reviews
    """
    ratings_matrix = []
    overall_ratings_list = []

    for review in reviews:
        # ratings = review['ratings']
        ratings_list = review['multi_ratings']

        # If there are not missing ratings, we add the ratings to the matrix
        # In other words, we are ignoring reviews with missing ratings
        if -1 not in ratings_list:
            ratings_matrix.append(ratings_list)
            overall_ratings_list.append(review['overall_rating'])

    return ratings_matrix, overall_ratings_list


def get_user_list(reviews, min_reviews):
    """
    Returns the list of users that have reviewed at least min_reviews hotels

    :param reviews: the list of reviews
    :param min_reviews: the minimum number of reviews
    :return: a list of user IDs
    """
    data_frame = DataFrame(reviews)
    column = 'user_id'
    counts = data_frame.groupby(column).size()
    filtered_counts = counts[counts >= min_reviews]
    # print(filtered_counts)
    num_users = len(filtered_counts)
    num_reviews = filtered_counts.sum()

    print('Number of users: %i' % num_users)
    print('Number of reviews: %i' % num_reviews)

    users = filtered_counts.index.get_level_values(1).tolist()
    return users


def get_groupby_list(reviews, column):
    """
    Groups the reviews by the given column and then returns all the distinct
    column values in a list

    :param reviews: the list of reviews
    :param column: the column which is going to be used to group the data
    :return: a list of all the distinct values of the given column in the
    reviews
    """
    data_frame = DataFrame(reviews)
    counts = data_frame.groupby(column).size()

    users = counts.index.get_level_values(1).tolist()
    return users


def get_item_list(reviews, min_reviews):
    """
    Returns the list of items that have at least min_reviews

    :param reviews: the list of reviews
    :param min_reviews: the minimum number of reviews
    :return: a list of item IDs
    """
    data_frame = DataFrame(reviews)
    column = 'offering_id'
    counts = data_frame.groupby(column).size()
    filtered_counts = counts[counts >= min_reviews]
    # print(filtered_counts)
    num_items = len(filtered_counts)
    num_reviews = filtered_counts.sum()

    print('Number of items: %i' % num_items)
    print('Number of reviews: %i' % num_reviews)

    items = filtered_counts.index.get_level_values(1).tolist()
    return items


def get_user_average_overall_rating(reviews, user_id, apply_filter=True):
    """
    Returns the average of the overall ratings that this user has given to
    every item he/she has reviewed

    :param reviews: a list of reviews
    :param user_id: the ID of the user
    :return: the average (or mean) of all the overall ratings that this has
    given to all the items he/she has reviewed
    """
    if apply_filter:
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    else:
        user_reviews = reviews

    ratings_sum = 0.
    ratings_count = len(user_reviews)

    for review in user_reviews:
        ratings_sum += float(review['overall_rating'])

    average_rating = float(ratings_sum) / float(ratings_count)

    return average_rating


def get_criteria_weights(reviews, user_id, apply_filter=True):
    """
    Obtains the weights for each of the criterion of the given user

    :param reviews: a list of all the available reviews
    :param user_id: the ID of the user
    :return: a list with the weights for each of the criterion of the given user
    """
    if apply_filter:
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    else:
        user_reviews = reviews

    ratings_matrix, overall_ratings_list = create_ratings_matrix(user_reviews)

    overall_ratings_matrix = numpy.vstack(
        [overall_ratings_list, numpy.ones(len(overall_ratings_list))]).T
    m, c = numpy.linalg.lstsq(overall_ratings_matrix, ratings_matrix)[0]

    return m


def get_significant_criteria(criteria_weights, ranges=None):
    """
    Returns (significant_criteria, cluster_name) where significant_criteria is a
    dictionary with the criteria that are significant and their values.
    cluster_name is the name of the cluster in which a user with the obtained
    significant criteria must belong

    :param criteria_weights: a list with the weights for each criterion
    :return: (significant_criteria, cluster_name) where significant_criteria is
    a dictionary with the criteria that are significant and their values.
    cluster_name is the name of the cluster in which a user with the obtained
    significant criteria must belong
    """
    rating_criteria = [
        'cleanliness',
        'location',
        'rooms',
        'service',
        # 'sleep_quality',
        'value'
    ]

    cluster_name = ''

    significant_criteria = {}
    if ranges is None:
        ranges = [(float("-inf"), float("inf"))]

    for index, value in enumerate(criteria_weights):
        if any(lower < value < upper for (lower, upper) in ranges):
            significant_criteria[rating_criteria[index]] = value
            cluster_name += '1'
        else:
            cluster_name += '0'

    return significant_criteria, cluster_name


def initialize_users(reviews, is_multi_criteria):
    """
    Builds a dictionary containing all the users in the reviews. Each user
    contains information about its average overall rating, the list of reviews
    that user has made, and the cluster the user belongs to

    :param reviews: the list of reviews
    :return: a dictionary with the users initialized, the keys of the
    dictionaries are the users' ID
    """
    user_ids = get_groupby_list(reviews, 'user_id')
    user_dictionary = {}

    for user_id in user_ids:
        user = User(user_id)
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
        user.average_overall_rating = get_user_average_overall_rating(
            user_reviews, user_id, apply_filter=False)
        user.item_ratings = get_user_item_ratings(user_reviews, user_id)
        user_dictionary[user_id] = user

        if is_multi_criteria:
            user.item_multi_ratings = get_user_item_multi_ratings(user_reviews, user_id)

    return user_dictionary


def initialize_cluster_users(reviews, significant_criteria_ranges=None):
    """
    Builds a dictionary containing all the users in the reviews. Each user
    contains information about its average overall rating, the list of reviews
    that user has made, and the cluster the user belongs to

    :param reviews: the list of reviews
    :return: a dictionary with the users initialized, the keys of the
    dictionaries are the users' ID
    """
    user_ids = get_groupby_list(reviews, 'user_id')
    user_dictionary = {}

    for user_id in user_ids:
        user = User(user_id)
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
        user.average_overall_rating = get_user_average_overall_rating(
            user_reviews, user_id, apply_filter=False)
        user.criteria_weights = get_criteria_weights(
            user_reviews, user_id, apply_filter=False)
        _, user.cluster = get_significant_criteria(
            user.criteria_weights, significant_criteria_ranges)
        user.item_ratings = get_user_item_ratings(user_reviews, user_id)
        user.item_multi_ratings = get_user_item_multi_ratings(user_reviews, user_id)
        user_dictionary[user_id] = user

    # print('Total users: %i' % len(user_ids))

    return user_dictionary


def get_user_item_ratings(reviews, user_id, apply_filter=False):
    """
    Returns a dictionary that contains the items that the given user has rated,
    where the key of the dictionary is the ID of the item and the value is the
    rating that user_id has given to that item

    :param reviews: a list of reviews
    :param user_id: the ID of the user
    :param apply_filter: a boolean that indicates if the reviews have to be
    filtered by user_id or not. In other word this boolean indicates if the list
    contains reviews from several users or not. If it does contains reviews from
    other users, those have to be removed
    :return: a dictionary with the items that the given user has rated
    """

    if apply_filter:
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    else:
        user_reviews = reviews

    if not user_reviews:
        return {}

    data_frame = DataFrame(user_reviews)
    column = 'offering_id'
    counts = data_frame.groupby(column).mean()

    items = counts.index.get_level_values(1).tolist()
    items_ratings = {}

    for item, mean in zip(items, counts['overall_rating']):
        items_ratings[item] = mean

    return items_ratings


def get_user_item_multi_ratings(reviews, user_id, apply_filter=False):
    """
    Returns a dictionary that contains the items that the given user has rated,
    where the key of the dictionary is the ID of the item and the value is the
    rating that user_id has given to that item. This function returns the
    multi-criteria ratings the user has made.

    :param reviews: a list of reviews
    :param user_id: the ID of the user
    :param apply_filter: a boolean that indicates if the reviews have to be
    filtered by user_id or not. In other word this boolean indicates if the list
    contains reviews from several users or not. If it does contains reviews from
    other users, those have to be removed
    :return: a dictionary with the items that the given user has rated
    """

    if apply_filter:
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    else:
        user_reviews = reviews

    user_multi_item_ratings = {}

    for item_id, item_reviews_it in itertools.groupby(
            user_reviews, operator.itemgetter('offering_id')):

        item_reviews = list(item_reviews_it)
        averaged_multi_ratings = [0] * len(item_reviews[0]['multi_ratings'])
        for review in item_reviews:

            averaged_rating = 0.
            rating_index = 0
            for rating in review['multi_ratings']:
                averaged_multi_ratings[rating_index] += rating / len(item_reviews)
                averaged_rating += rating
                rating_index += 1

        user_multi_item_ratings[item_id] = averaged_multi_ratings

    return user_multi_item_ratings


def get_five_star_hotels_from_user(user_reviews, min_value):
    """
    Returns the list of hotels that this user has reviewed with an average
    overall rating higher than min_value

    :param user_reviews: the reviews the user has made
    :param min_value: the minimum value for the average overall rating that this
    user has given to a hotel
    :return: the list of hotels that this user has reviewed with an average
    overall rating higher than min_value
    """
    data_frame = DataFrame(user_reviews)
    column = 'offering_id'
    counts = data_frame.groupby(column).mean()
    filtered_counts = counts[counts['overall_rating'] >= min_value]

    # print(filtered_counts)

    items = filtered_counts.index.get_level_values(1).tolist()
    return items


def get_common_items(user_dictionary, user1, user2):
    items_user1 = set(user_dictionary[user1].item_ratings.keys())
    items_user2 = set(user_dictionary[user2].item_ratings.keys())

    common_items = items_user1.intersection(items_user2)

    return common_items


def get_user_ratings(user_dictionary, user, items):

    ratings = []

    for item in items:
        ratings.append(get_rating(user_dictionary, user, item))

    return ratings


def get_rating(user_dictionary, user, item):
    if item in user_dictionary[user].item_ratings:
        return user_dictionary[user].item_ratings[item]
    return None


def get_user_multi_ratings(user_dictionary, user, items):

    ratings = []

    for item in items:
        ratings.append(get_multi_ratings(
            user_dictionary, user, item))

    return ratings


def get_multi_ratings(user_dictionary, user, item):
    if item in user_dictionary[user].item_multi_ratings:
        return user_dictionary[user].item_multi_ratings[item]
    return None


def get_matrix_column(matrix, i):
    return [row[i] for row in matrix]


def main():
    # reviews = pre_process_reviews()
    # save_dictionary_list_to_file(reviews, '/Users/fpena/tmp/filtered_reviews.json')
    reviews = ETLUtils.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
    data_frame = DataFrame(reviews)
    column = 'offering_id'
    groupby = data_frame.groupby(column)
    counts = groupby.mean()
    print(counts)

    items = counts.index.get_level_values(1).tolist()

    for item, mean in zip(items, counts['overall_rating']):
        print(item, mean)

    # print(get_item_list(reviews, 2))
    # print(len(reviews))
    # initialize_users(reviews, 10)
    pass


# start_time = time.time()
# main()
# end_time = time.time() - start_time
# print("--- %s seconds ---" % end_time)
