from collections import Counter
import math
import numpy

from nlp import nlp_utils
from utils.constants import Constants

__author__ = 'fpena'


def get_review_metrics(record):
    """
    Returns a list with the metrics of a review. This list is composed
    in the following way: [log(num_sentences + 1), log(num_words + 1),
    log(num_past_verbs + 1), log(num_verbs + 1),
    (log(num_past_verbs + 1) / log(num_verbs + 1))

    :type record: dict
    :param record: the review that wants to be analyzed, it should contain the
    text of the review
    :rtype: list[float]
    :return: a list with numeric metrics
    """
    review_text = record[Constants.TEXT_FIELD]
    log_sentences = math.log(len(nlp_utils.get_sentences(review_text)) + 1)
    log_words = math.log(len(nlp_utils.get_words(review_text)) + 1)
    # log_time_words = math.log(len(self.get_time_words(review.text)) + 1)
    tagged_words = record[Constants.POS_TAGS_FIELD]
    counts = Counter(tag for word, tag, lemma in tagged_words)
    # print(counts)
    log_past_verbs = math.log(counts['VBD'] + 1)
    log_verbs = math.log(nlp_utils.count_verbs(counts) + 1)
    log_personal_pronouns = math.log(counts['PRP'] + 1)
    # log_sentences = float(len(get_sentences(review.text)) + 1)
    # log_words = float(len(get_words(review.text)) + 1)
    # log_time_words = float(len(self.get_time_words(review.text)) + 1)
    # tagged_words = review.tagged_words
    # counts = Counter(tag for word, tag in tagged_words)
    # log_past_verbs = float(counts['VBD'] + 1)
    # log_verbs = float(count_verbs(counts) + 1)
    # log_personal_pronouns = float(counts['PRP'] + 1)

    # This ensures that when log_verbs = 0 the program won't crash
    if log_verbs == 0:
        past_verbs_ratio = 0
    else:
        past_verbs_ratio = log_past_verbs / log_verbs
    # This ensures that when log_verbs = 0 the program won't crash
    if log_words == 0:
        verbs_ratio = 0
        past_verbs_ratio2 = 0
        personal_pronouns_ratio = 0
    else:
        verbs_ratio = log_verbs / log_words
        past_verbs_ratio2 = log_past_verbs / log_words
        # time_words_ratio = log_time_words / log_words
        personal_pronouns_ratio = log_personal_pronouns / log_words

    result = [log_words, log_past_verbs, log_verbs, past_verbs_ratio, verbs_ratio, log_personal_pronouns, personal_pronouns_ratio, past_verbs_ratio2]#, log_time_words, time_words_ratio]
    # print(result)
    # result = [log_sentences]
    # result = [log_personal_pronouns, log_past_verbs, log_words]
    # result = [time_words_ratio, personal_pronouns_ratio]
    # result = [log_past_verbs]
    # result = [log_verbs]
    # result = [past_verbs_ratio]
    # result = [past_verbs_ratio2]
    # result = [log_personal_pronouns]
    # result = [personal_pronouns_ratio]
    # result = [log_time_words]
    # result = [time_words_ratio]

    return numpy.array(result)


def normalize_matrix_by_columns(matrix, min_values=None, max_values=None):
    """

    :type matrix: numpy.array
    """
    if max_values is None:
        max_values = matrix.max(axis=0)
    if min_values is None:
        min_values = matrix.min(axis=0)

    # print('max values', max_values)
    # print('min values', min_values)

    for index in range(matrix.shape[1]):
        matrix[:, index] -= min_values[index]
        matrix[:, index] /= (max_values[index] - min_values[index])
