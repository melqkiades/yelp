from collections import Counter
import cPickle as pickle

import numpy as np
from sklearn.cluster import KMeans

from etl import ETLUtils
from topicmodeling.context import review_metrics_extractor
from topicmodeling.context.review import Review

__author__ = 'fpena'


NUM_FEATURES = 2


def cluster_reviews(reviews):
    """
    Classifies a list of reviews into specific and generic. Returns a list of
    integer of the same size as the list of reviews, in which each position of
    the list contains a 0 if that review is specific or a 1 if that review is
    generic.

    :param reviews: a list of reviews. Each review must contain the text of the
    review and the part-of-speech tags for every word
    :type reviews: list[Review]
    :return a list of integer of the same size as the list of reviews, in which
    each position of the list contains a 0 if that review is specific or a 1 if
    that review is generic
    """

    metrics = np.zeros((len(reviews), NUM_FEATURES))

    for index in range(len(reviews)):
        metrics[index] =\
            review_metrics_extractor.get_review_metrics(reviews[index])
    review_metrics_extractor.normalize_matrix_by_columns(metrics)

    k_means = KMeans(n_clusters=2)
    k_means.fit(metrics)
    labels = k_means.labels_

    record_clusters = split_list_by_labels(metrics, labels)
    cluster0_sum = reduce(lambda x, y: x + sum(y), record_clusters[0], 0)
    cluster1_sum = reduce(lambda x, y: x + sum(y), record_clusters[1], 0)

    if cluster0_sum < cluster1_sum:
        # If the cluster 0 contains the generic review we invert the tags
        labels = [1 if element == 0 else 0 for element in labels]

    return labels


def split_list_by_labels(lst, labels):
    """
    Receives a list of objects and a list of labels (each label is an integer
    number) and returns a matrix in which all the elements have been
    grouped by label. For instance if we have lst = ['a', 'b', 'c', 'd', 'e']
    and labels = [0, 0, 1, 1, 0] then the result of calling this function would
    be matrix = [['a', 'b', 'e'], ['c', 'd']]. This is particularly useful when
    we call matrix[0] = ['a', 'b', 'e'] or matrix[1] = ['c', 'd']

    :type lst: numpy.array
    :param lst: a list of objects
    :type labels: list[int]
    :param labels: a list of integer with the label for each element of lst
    :rtype: list[list[]]
    :return:
    """
    matrix = []

    for index in range(max(labels) + 1):
        matrix.append([])

    for index in range(len(labels)):
        element = lst[index]
        matrix[labels[index]].append(element)

    return matrix


def get_stats_from_reviews(reviews):

    records = np.zeros((len(reviews), 5))

    for index in range(len(reviews)):
        records[index] = count_review_info(reviews[index])

    max_values = records.max(axis=0)
    min_values = records.min(axis=0)
    mean_values = records.mean(axis=0)

    stats = {
        'total_reviews': len(reviews),
        'sentences': {'max': max_values[0], 'min': min_values[0], 'mean': mean_values[0]},
        'words': {'max': max_values[1], 'min': min_values[1], 'mean': mean_values[1]},
        'past_verbs': {'max': max_values[2], 'min': min_values[2], 'mean': mean_values[2]},
        'verbs': {'max': max_values[3], 'min': min_values[3], 'mean': mean_values[3]},
        'ratio': {'max': max_values[4], 'min': min_values[4], 'mean': mean_values[4]},
    }

    return stats


def count_review_info(review):
    num_sentences = len(review_metrics_extractor.get_sentences(review.text))
    num_words = len(review_metrics_extractor.get_words(review.text))
    tagged_words = review.tagged_words
    counts = Counter(tag for word, tag in tagged_words)
    num_past_verbs = float(counts['VBD'])
    num_verbs = review_metrics_extractor.count_verbs(counts)

    # This ensures that when log_verbs = 0 the program won't crash
    if num_verbs == 0:
        verbs_ratio = 0
    else:
        verbs_ratio = num_past_verbs / num_verbs

    result = [num_sentences, num_words, num_past_verbs, num_verbs, verbs_ratio]

    # print('ratio', verbs_ratio, '\tpast verbs', num_past_verbs, 'verbs', num_verbs)

    return np.array(result)


def main():
    # my_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_hotel_reviews.json'
    my_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_restaurant_reviews.json'
    my_records = ETLUtils.load_json_file(my_file)
    # my_reviews = []
    # my_index = 0
    #
    # print("records:", len(my_records))
    #
    # for record in my_records:
    #     my_index += 1
    #     my_reviews.append(Review(record['text']))
    #     print('index', my_index)

    # binary_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_hotel_reviews.pkl'
    binary_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_restaurant_reviews.pkl'
    # with open(binary_reviews_file, 'wb') as write_file:
    #     pickle.dump(my_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    with open(binary_reviews_file, 'rb') as read_file:
        my_reviews = pickle.load(read_file)

    cluster_labels = cluster_reviews(my_reviews)
    specific_records = split_list_by_labels(my_records, cluster_labels)[0]
    generic_records = split_list_by_labels(my_records, cluster_labels)[1]
