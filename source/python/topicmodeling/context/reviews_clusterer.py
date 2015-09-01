from collections import Counter
import string
import math
from nltk import tokenize
import nltk
import numpy as np
import cPickle as pickle
from sklearn.cluster import KMeans
from topicmodeling.context.review import Review

__author__ = 'fpena'


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

    records = np.zeros((len(reviews), 5))

    for index in range(len(reviews)):
        records[index] = get_review_metrics(reviews[index])
    normalize_matrix_by_columns(records)

    k_means = KMeans(n_clusters=2)
    k_means.fit(records)
    labels = k_means.labels_

    record_clusters = split_list_by_labels(records, labels)
    cluster0_sum = reduce(lambda x, y: x + sum(y), record_clusters[0], 0)
    cluster1_sum = reduce(lambda x, y: x + sum(y), record_clusters[1], 0)

    if cluster0_sum < cluster1_sum:
        # If the cluster 0 contains the generic review we invert the tags
        labels = [1 if element == 0 else 0 for element in labels]

    return labels


def normalize_matrix_by_columns(matrix):
    """

    :type matrix: numpy.array
    """
    max_values = matrix.max(axis=0)

    for index in range(matrix.shape[1]):
        matrix[:, index] /= max_values[index]


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
    :rtype: list[list[int]]
    :return:
    """
    matrix = []

    for index in range(max(labels) + 1):
        matrix.append([])

    for index in range(len(labels)):
        element = lst[index]
        matrix[labels[index]].append(element)

    return matrix


def get_review_metrics(review):
    """
    Returns a list with the metrics of a review. This list is composed
    in the following way: [log(num_sentences + 1), log(num_words + 1),
    log(num_past_verbs + 1), log(num_verbs + 1),
    (log(num_past_verbs + 1) / log(num_verbs + 1))

    :type review: Review
    :param review: the review that wants to be analyzed, it should contain the
    text of the review and the part-of-speech tags for every word in the review
    :rtype: list[float]
    :return: a list with numeric metrics
    """
    log_sentence = math.log(count_sentences(review.text) + 1)
    log_word = math.log(count_words(review.text) + 1)
    tagged_words = review.tagged_words
    counts = Counter(tag for word, tag in tagged_words)
    log_past_verbs = math.log(counts['VBD'] + 1)
    log_verbs = math.log(count_verbs(counts) + 1)

    # This ensures that when log_verbs = 0 the program won't crash
    if log_verbs == 0:
        verbs_ratio = 0
    else:
        verbs_ratio = log_past_verbs / log_verbs

    result = [log_sentence, log_word, log_past_verbs, log_verbs, verbs_ratio]

    return np.array(result)


def count_sentences(text):
    """
    Returns the number of sentences there are in the given text

    :type text: str
    :param text: just a text
    :rtype: int
    :return: the number of sentences there are in the given text
    """
    return len(tokenize.sent_tokenize(text))


def count_words(text):
    """
    Returns the number of words there are in the given text

    :type text: str
    :param text: just a text. It must be in english.
    :rtype: int
    :return: the number of words there are in the given text
    """
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)

    words = []

    for sentence in sentences:
        words.extend(
            [word.strip(string.punctuation) for word in sentence.split()])
    return len(words)


def count_verbs(tags_count):
    """
    Receives a dictionary with part-of-speech tags as keys and counts as values,
    returns the total number of verbs that appear in the dictionary

    :type tags_count: dict
    :param tags_count: a dictionary with part-of-speech tags as keys and counts
    as values
    :rtype : int
    :return: the total number of verbs that appear in the dictionary
    """

    total_verbs =\
        tags_count['VB'] + tags_count['VBD'] + tags_count['VBG'] +\
        tags_count['VBN'] + tags_count['VBP'] + tags_count['VBZ']

    return total_verbs


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
    num_sentences = count_sentences(review.text)
    num_words = count_words(review.text)
    tagged_words = review.tagged_words
    counts = Counter(tag for word, tag in tagged_words)
    num_past_verbs = float(counts['VBD'])
    num_verbs = count_verbs(counts)

    # This ensures that when log_verbs = 0 the program won't crash
    if num_verbs == 0:
        verbs_ratio = 0
    else:
        verbs_ratio = num_past_verbs / num_verbs

    result = [num_sentences, num_words, num_past_verbs, num_verbs, verbs_ratio]

    # print('ratio', verbs_ratio, '\tpast verbs', num_past_verbs, 'verbs', num_verbs)

    return np.array(result)


# my_file = '/Users/fpena/tmp/reviews_restaurant_shuffled.pkl'
# # my_file = '/Users/fpena/tmp/reviews_hotel_shuffled.pkl'
# # my_file = '/Users/fpena/tmp/reviews_spa.pkl'
# with open(my_file, 'rb') as read_file:
#     my_reviews = pickle.load(read_file)
#
# print(get_stats_from_reviews(my_reviews))
