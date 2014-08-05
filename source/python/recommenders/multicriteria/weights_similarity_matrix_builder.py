from etl import similarity_calculator

__author__ = 'fpena'


def calculate_users_similarity(user_dictionary, user_id1, user_id2, similarity_metric='euclidean'):
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

    return similarity_calculator.calculate_similarity(user_weights1, user_weights2, similarity_metric)


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
                calculate_users_similarity(user_dictionary, user1, user2, similarity_metric)

    return user_similarity_matrix
