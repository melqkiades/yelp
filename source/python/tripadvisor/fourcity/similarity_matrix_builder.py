from etl import similarity_calculator

__author__ = 'fpena'


def build_similarity_matrix(user_ids, user_dictionary, similarity_metric='euclidean'):
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
                calculate_similarity(user_dictionary, user1, user2, similarity_metric)

    return user_similarity_matrix


def get_common_items(user_dictionary, user1, user2):
    items_user1 = set(user_dictionary[user1].item_ratings.keys())
    items_user2 = set(user_dictionary[user2].item_ratings.keys())

    common_items = items_user1.intersection(items_user2)

    return common_items


def calculate_similarity(user_dictionary, user1, user2, similarity_metric):
    common_items = get_common_items(user_dictionary, user1, user2)

    if not common_items:
        return None

    user1_ratings = extract_user_ratings(user_dictionary, user1, common_items)
    user2_ratings = extract_user_ratings(user_dictionary, user2, common_items)

    similarity_value = similarity_calculator.calculate_similarity(
        user1_ratings, user2_ratings, similarity_metric)

    return similarity_value


def extract_user_ratings(user_dictionary, user, items):

    ratings = []

    for item in items:
        ratings.append(get_rating(user_dictionary, user, item))

    return ratings


def get_rating(user_dictionary, user, item):
    if item in user_dictionary[user].item_ratings:
        return user_dictionary[user].item_ratings[item]
    return None
