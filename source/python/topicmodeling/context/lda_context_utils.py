import time

import nltk
from gensim import corpora
from gensim.models import ldamodel, LdaMulticore
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import numpy
from etl import ETLUtils
from utils.constants import Constants

__author__ = 'fpena'


def build_topic_model_from_corpus(corpus, dictionary):
    """
    Builds a topic model with the given corpus and dictionary.
    The model is built using Latent Dirichlet Allocation

    :type corpus list
    :parameter corpus: a list of bag of words, each bag of words represents a
    document
    :type dictionary: gensim.corpora.Dictionary
    :parameter dictionary: a Dictionary object that contains the words that are
    permitted to belong to the document, words that are not in this dictionary
    will be ignored
    :rtype: gensim.models.ldamodel.LdaModel
    :return: an LdaModel built using the reviews contained in the records
    parameter
    """

    # numpy.random.seed(0)
    if Constants.LDA_MULTICORE:
        print('%s: lda multicore' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        topic_model = LdaMulticore(
            corpus, id2word=dictionary,
            num_topics=Constants.TOPIC_MODEL_NUM_TOPICS,
            passes=Constants.TOPIC_MODEL_PASSES,
            iterations=Constants.TOPIC_MODEL_ITERATIONS,
            workers=Constants.NUM_CORES - 1)
    else:
        print('%s: lda monocore' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        topic_model = ldamodel.LdaModel(
            corpus, id2word=dictionary,
            num_topics=Constants.TOPIC_MODEL_NUM_TOPICS,
            passes=Constants.TOPIC_MODEL_PASSES,
            iterations=Constants.TOPIC_MODEL_ITERATIONS)

    return topic_model


def update_reviews_with_topics(topic_model, corpus_list, reviews):
    """

    :type minimum_probability: float
    :param minimum_probability:
    :type topic_model: LdaModel
    :param topic_model:
    :type corpus_list: list
    :param reviews:
    """
    # print('reviews length', len(reviews))

    for review, corpus in zip(reviews, corpus_list):
        review[Constants.TOPICS_FIELD] =\
            topic_model.get_document_topics(corpus)

        non_zero_topics = [topic[0] for topic in review[Constants.TOPICS_FIELD]]

        for topic_index in range(Constants.TOPIC_MODEL_NUM_TOPICS):
            if topic_index not in non_zero_topics:
                review[Constants.TOPICS_FIELD].insert(
                    topic_index, [topic_index, 0.0])


def calculate_topic_weighted_frequency(topic, reviews):
    """

    :type topic: int
    :param topic:
    :type reviews: list[dict]
    :param reviews:
    :return:
    """
    num_reviews = 0.0

    for review in reviews:
        for review_topic in review[Constants.TOPICS_FIELD]:
            if topic == review_topic[0]:
                if Constants.TOPIC_WEIGHTING_METHOD == 'binary':
                    num_reviews += 1
                elif Constants.TOPIC_WEIGHTING_METHOD == 'probability':
                    num_reviews += review_topic[1]
                else:
                    raise ValueError('Topic weighting method not recognized')

    return num_reviews / len(reviews)


def calculate_topic_weighted_frequency_complete(topic, reviews):
    """

    :type topic: int
    :param topic:
    :type reviews: list[dict]
    :param reviews:
    :return:
    """
    review_frequency = 0.0
    log_words_frequency = 0.0
    log_past_verbs_frequency = 0.0

    for review in reviews:
        for review_topic in review[Constants.TOPICS_FIELD]:
            if topic == review_topic[0]:
                log_words_frequency += review['log_words']
                log_past_verbs_frequency += review['log_past_verbs']
                if Constants.TOPIC_WEIGHTING_METHOD == 'binary':
                    review_frequency += 1
                elif Constants.TOPIC_WEIGHTING_METHOD == 'probability':
                    review_frequency += review_topic[1]
                else:
                    raise ValueError('Topic weighting method not recognized')

    results = {
        'review_frequency': review_frequency / len(reviews),
        'log_words_frequency': log_words_frequency / len(reviews),
        'log_past_verbs_frequency': log_past_verbs_frequency / len(reviews)
    }

    return results


def create_bag_of_words(document_list):
    """
    Creates a bag of words representation of the document list given. It removes
    the punctuation and the stop words.

    :type document_list: list[str]
    :param document_list:
    :rtype: list[list[str]]
    :return:
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tagger = nltk.PerceptronTagger()
    cached_stop_words = set(stopwords.words("english"))
    cached_stop_words |= {
        't', 'didn', 'doesn', 'haven', 'don', 'aren', 'isn', 've', 'll',
        'couldn', 'm', 'hasn', 'hadn', 'won', 'shouldn', 's', 'wasn',
        'wouldn'}
    body = []
    processed = []

    for i in range(0, len(document_list)):
        body.append(document_list[i].lower())

    for entry in body:
        row = tokenizer.tokenize(entry)
        tagged_words = tagger.tag(row)

        nouns = []
        for tagged_word in tagged_words:
            if tagged_word[1].startswith('NN'):
                nouns.append(tagged_word[0])

        nouns = [word for word in nouns if word not in cached_stop_words]
        processed.append(nouns)

    return processed


def get_user_item_reviews(records, user_id, apply_filter=False):

    if apply_filter:
        user_records = ETLUtils.filter_records(records, 'user_id', [user_id])
    else:
        user_records = records

    if not user_records:
        return {}

    items_reviews = {}

    for record in user_records:
        items_reviews[record['offering_id']] = record['text']

    return items_reviews


def get_user_item_contexts(
        records, lda_model, user_id, apply_filter=False,
        minimum_probability=None):

    if apply_filter:
        user_records = ETLUtils.filter_records(records, 'user_id', [user_id])
    else:
        user_records = records

    if not user_records:
        return {}

    items_reviews = {}

    for record in user_records:
        review_text = record['text']
        context = get_topic_distribution(
            review_text, lda_model, minimum_probability)
        items_reviews[record['offering_id']] = context

    return items_reviews


def get_topic_distribution(record, lda_model, dictionary, minimum_probability,
                           sampling_method=None, max_words=None):
    """

    :type record: dict
    :type lda_model: LdaModel
    :type minimum_probability: float
    :param sampling_method: a float in the range [0,1] that
    indicates the proportion of text that should be sampled from the review.
    It can also take the string value of 'max', indicating that only the
    word with the highest probability from the topic will be sampled
     text. If None then all the review text is taken
    :param max_words: is the set of words with maximum probability for each
    contextual topic
    """
    # review_bow = [record[Constants.BOW_FIELD]]
    # review_bow =\
    #     sample_bag_of_words(review_bow, sampling_method, max_words)

    # corpus = dictionary.doc2bow(review_bow[0])
    corpus = record[Constants.CORPUS_FIELD]
    lda_corpus = lda_model.get_document_topics(
        corpus, minimum_probability=minimum_probability)

    topic_distribution = numpy.zeros(lda_model.num_topics)
    for pair in lda_corpus:
        topic_distribution[pair[0]] = pair[1]

    return topic_distribution


def sample_bag_of_words(review_bow, sampling_method, max_words=None):
    """
    Samples a list of strings containing a bag of words using the given
    sampling method. The sampling is always done without replacement.

    :type review_bow: list[str]
    :param review_bow: the bag of words to sample
    :param sampling_method: a float in the range [0,1] that
    indicates the proportion of text that should be sampled from the review.
    It can also take the string value of 'max', indicating that only the
    word with the highest probability from the topic will be sampled
     text. If None then the origianl review_bow list is returned
    :param max_words: is the set of words with maximum probability for each
    contextual topic
    """

    if sampling_method is None or len(review_bow) == 0:
        return review_bow

    if sampling_method == 'max':
        bow_set = set(review_bow)
        words_set = set(max_words)
        review_bow = list(bow_set.intersection(words_set))
        return review_bow
    elif 0.0 <= sampling_method <= 1.0:
        num_words = int(sampling_method * len(review_bow))
        review_bow = numpy.random.choice(review_bow, num_words, replace=False)

        return review_bow


def main():

    review = 'The quick brown fox jumps over the lazy dog'
    review_bow = review.split(' ')
    print('original number of words: %d' % len(review_bow))

    sampled_bows = sample_bag_of_words(review_bow, 1.0)
    print('number of words when sample = 1.0: %d' % len(sampled_bows))
    print(sampled_bows)

    sampled_bows = sample_bag_of_words(review_bow, 0.5)
    print('number of words when sample = 0.5: %d' % len(sampled_bows))
    print(sampled_bows)

    max_words = ['quick', 'lazy']
    sampled_bows = sample_bag_of_words(review_bow, 'max', max_words)
    print('number of words when sample = max_words: %d' % len(sampled_bows))
    print(sampled_bows)


# main()
