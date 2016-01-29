from collections import Counter
import math
import string
import nltk
import numpy
from topicmodeling.context.review import Review

__author__ = 'fpena'


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
    log_sentences = math.log(len(get_sentences(review.text)) + 1)
    log_words = math.log(len(get_words(review.text)) + 1)
    # log_time_words = math.log(len(self.get_time_words(review.text)) + 1)
    tagged_words = review.tagged_words
    counts = Counter(tag for word, tag in tagged_words)
    log_past_verbs = math.log(counts['VBD'] + 1)
    log_verbs = math.log(count_verbs(counts) + 1)
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
        time_words_ratio = 0
        personal_pronouns_ratio = 0
    else:
        verbs_ratio = log_verbs / log_words
        past_verbs_ratio2 = log_past_verbs / log_words
        # time_words_ratio = log_time_words / log_words
        personal_pronouns_ratio = log_personal_pronouns / log_words

    # result = [log_sentences, log_words, log_past_verbs, log_verbs, past_verbs_ratio, log_personal_pronouns, personal_pronouns_ratio, past_verbs_ratio2]#, log_time_words, time_words_ratio]
    # result = [log_sentences]
    result = [log_words, log_past_verbs]
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

    print('max values', max_values)
    print('min values', min_values)

    for index in range(matrix.shape[1]):
        matrix[:, index] -= min_values[index]
        matrix[:, index] /= (max_values[index] - min_values[index])


def build_reviews(records):
    reviews = []
    count = 0
    for record in records:
        review = Review(record['text'])
        review.user_id = record['user_id']
        review.item_id = record['business_id']
        review.rating = record['stars']
        reviews.append(review)
        count += 1
        print('count: %d/%d\r' % (count, len(records))),

    return reviews


def get_sentences(text):
    """
    Returns a list with the sentences there are in the given text

    :type text: str
    :param text: just a text
    :rtype: list[str]
    :return: a list with the sentences there are in the given text
    """
    return nltk.tokenize.sent_tokenize(text)


def get_words(text):
    """
    Splits the given text into words and returns them

    :type text: str
    :param text: just a text. It must be in english.
    :rtype: list[str]
    :return: a list with the words there are in the given text
    """
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)

    words = []

    for sentence in sentences:
        words.extend(
            [word.strip(string.punctuation) for word in sentence.split()])
    return words


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
