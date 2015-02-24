import numpy
from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


# records = ETLUtils.load_csv_file('/Users/fpena/UCC/Thesis/datasets/ml-1m.csv', '|')
# records = ETLUtils.load_csv_file('/Users/fpena/UCC/Thesis/datasets/ml-10m.csv', '|')

def get_ml_100K_dataset():
    # records = ETLUtils.load_csv_file('/Users/fpena/tmp/bpmf/ml-1k.csv', '\t')
    records = ETLUtils.load_csv_file('/Users/fpena/tmp/bpmf/ml-100k.csv', '\t')
    # records = ETLUtils.load_csv_file('/Users/fpena/UCC/Thesis/datasets/uncompressed/ml-100k.csv', '\t')
    for record in records:
        record['overall_rating'] = float(record['overall_rating'])
    return records


def get_ml_1m_dataset():
    records = ETLUtils.load_csv_file('/Users/fpena/UCC/Thesis/datasets/uncompressed/ml-1m.csv', '|')
    for record in records:
        record['overall_rating'] = float(record['overall_rating'])
    return records


def clean_reviews(reviews):
    """
    Returns a copy of the original reviews list with only that are useful for
    recommendation purposes

    :param reviews: a list of reviews
    :return: a copy of the original reviews list with only that are useful for
    recommendation purposes
    """
    # print('Finished remove_users_with_low_reviews')
    filtered_reviews = extractor.remove_items_with_low_reviews(reviews, 100)
    # print('Finished remove_single_review_hotels')
    filtered_reviews = extractor.remove_users_with_low_reviews(filtered_reviews, 200)
    # print('Finished remove_users_with_low_reviews')
    print('Number of reviews', len(filtered_reviews))
    return filtered_reviews


def reviews_to_numpy_matrix(reviews):

    reviews_matrix = []

    for review in reviews:
        row = [
            int(review['user_id']),
            int(review['offering_id']),
            int(review['overall_rating'])
        ]
        reviews_matrix.append(row)

    numpy_matrix = numpy.array(reviews_matrix)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    numpy_matrix[:, (0, 1)] = numpy_matrix[:, (0, 1)] - 1

    print("max user id", max(numpy_matrix[:, 0]))
    print("max item id", max(numpy_matrix[:, 1]))

    return numpy_matrix


def get_mean_rating(numpy_ratings):

    return numpy.mean(numpy_ratings[:, 2])


def get_num_users(numpy_ratings):

    # return len(numpy.unique(numpy_ratings[:, 0]))
    return max(numpy_ratings[:, 0]) + 1


def get_num_items(numpy_ratings):

    # return len(numpy.unique(numpy_ratings[:, 1]))
    return max(numpy_ratings[:, 1]) + 1

# clean_reviews(get_ml_100K_dataset())
# records = get_ml_100K_dataset()
# records = get_ml_1m_dataset()
# print(records[0])
# print(records[1])
# print(records[10])
# print(records[100])
# print(records[0])
#
# matrix = reviews_to_numpy_matrix(records)
#
# print(matrix[0])
# print(matrix[1])
# print(matrix[10])
# print(matrix[100])
# print(len(numpy.unique(matrix[:, 0])))
# print(len(numpy.unique(matrix[:, 1])))

# print "max user id", max(matrix[:, 0])
# print "max item id", max(matrix[:, 1])
