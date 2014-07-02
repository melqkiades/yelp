import numpy
from pandas import DataFrame
import time
from etl import ETLUtils
from tripadvisor.fourcity.user import User


__author__ = 'fpena'


def get_dictionary_subfield(dataDict, mapList):
    return reduce(lambda d, k: d[k], mapList, dataDict)


def extract_fields(reviews):

    ratings_criteria = [
        'cleanliness',
        'location',
        'rooms',
        'service',
        # 'sleep_quality',
        'value'
    ]

    for review in reviews:
        # review['user_id'] = get_dictionary_subfield(review, ['author', 'id'])
        review['user_id'] = review['author']['id']
        review['overall_rating'] = review['ratings']['overall']

        ratings = review['ratings']
        for criterion in ratings_criteria:
            if criterion in ratings:
                review[criterion + '_rating'] = review['ratings'][criterion]


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


def remove_single_review_hotels(reviews):
    """
    TODO: Missing, this method is just a stub
    Returns a copy of the original reviews list without the reviews of hotels
    that just have been reviewed once

    :param reviews: a list of reviews
    :return: a copy of the original reviews list without the reviews of hotels
    that just have been reviewed once
    """
    items = get_item_list(reviews, 2)
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
    expected_criteria = [
        'cleanliness',
        'location',
        'rooms',
        'service',
        # 'sleep_quality',
        'value'
    ]
    expected_criteria = set(expected_criteria)
    actual_criteria = set(review['ratings'])
    return expected_criteria.issubset(actual_criteria)


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
    # filtered_reviews = remove_single_review_hotels(filtered_reviews)
    print('Finished remove_single_review_hotels')
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
    ETLUtils.drop_fields(['author'], reviews)
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

    missing_count = 0

    for review in reviews:
        ratings = review['ratings']
        ratings_list = []

        rating_criteria = [
            'cleanliness',
            'location',
            'rooms',
            'service',
            # 'sleep_quality',
            'value'
        ]
        contains_missing_rating = 0

        for criterion in rating_criteria:
            if criterion in ratings:
                ratings_list.append(ratings[criterion])

        missing_count += contains_missing_rating

        # If there are not missing ratings, we add the ratings to the matrix
        # In other words, we are ignoring reviews with missing ratings
        if not contains_missing_rating:
            ratings_matrix.append(ratings_list)
            overall_ratings_list.append(ratings['overall'])
            review['ratings_list'] = ratings_list
            # review['overall_rating'] = ratings['overall']

    # print(missing_count)
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
        ratings_sum += review['overall_rating']

    average_rating = float(ratings_sum) / float(ratings_count)

    return average_rating


def get_criteria_weights(reviews, user_id, apply_filter=True):
    """
    Obtains the weights for each of the criterion of the given user

    :param reviews: a list of all the available reviews
    :param user_id: the ID of the user
    :return: a list with the weights for each of the criterion of the given user
    """
    # filtered_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    if apply_filter:
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    else:
        user_reviews = reviews

    ratings_matrix, overall_ratings_list = create_ratings_matrix(user_reviews)

    overall_ratings_matrix = numpy.vstack(
        [overall_ratings_list, numpy.ones(len(overall_ratings_list))]).T
    m, c = numpy.linalg.lstsq(overall_ratings_matrix, ratings_matrix)[0]

    return m


def get_significant_criteria(criteria_weights):
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
    for index, value in enumerate(criteria_weights):
        # if (0.8 < value < 1.2) or (-1.2 < value < -0.8):
        # if (0.7 < value < 1.3) or (-1.3 < value < -0.7):
        if (0.5 < value < 1.5) or (-1.5 < value < -0.5):
        # if (0.1 < value < 1.9) or (-1.1 < value < -0.9):
        # if True:
            significant_criteria[rating_criteria[index]] = value
            cluster_name += '1'
        else:
            cluster_name += '0'

    return significant_criteria, cluster_name


def initialize_users(reviews, min_reviews):

    user_ids = get_user_list(reviews, min_reviews)
    user_dictionary = {}

    for user_id in user_ids:
        user = User(user_id)
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
        user.average_overall_rating = get_user_average_overall_rating(
            user_reviews, user_id, apply_filter=False)
        user.criteria_weights = get_criteria_weights(
            user_reviews, user_id, apply_filter=False)
        _, user.cluster = get_significant_criteria(user.criteria_weights)
        user_dictionary[user_id] = user

    # for user_id, user in user_dictionary.iteritems():
    #     print('ID: %s\tOverall average: %f\tCluster: %s' % (user.user_id, user.average_overall_rating, user.cluster))
        # print(user.user_id)
        # print('Average rating: ' + str(user.average_overall_rating))
        # print('Cluster: ' + str(user.cluster))
    print('Total users: %i' % len(user_ids))

    return user_dictionary


def get_five_star_hotels_from_user(user_reviews):

    data_frame = DataFrame(user_reviews)
    column = 'offering_id'
    counts = data_frame.groupby(column).mean()
    filtered_counts = counts[counts['overall_rating'] >= 4.5]

    # print(filtered_counts)

    items = filtered_counts.index.get_level_values(1).tolist()
    return items


def main():
    reviews = pre_process_reviews()
    # print(get_item_list(reviews, 2))
    print(len(reviews))
    # initialize_users(reviews, 10)
    pass


start_time = time.time()
main()
end_time = time.time() - start_time
print("--- %s seconds ---" % end_time)

