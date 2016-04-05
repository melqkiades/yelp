from gensim import corpora
from gensim.models import ldamodel
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import numpy
from etl import ETLUtils
from utils.constants import Constants

__author__ = 'fpena'


def update_reviews_with_topics(topic_model, corpus_list, reviews):
    """

    :type topic_model: LdaModel
    :param topic_model:
    :type reviews: list[dict]
    :param reviews:
    """
    # print('reviews length', len(reviews))

    for review, corpus in zip(reviews, corpus_list):
        review[Constants.TOPICS_FIELD] = topic_model.get_document_topics(corpus)


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
                num_reviews += 1

    return num_reviews / len(reviews)


def discover_topics(text_reviews, num_topics):

    processed = create_bag_of_words(text_reviews)

    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(2, 0.6)

    # dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in processed]

    # I can print out the documents and which is the most probable topics for each doc.
    lda_model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
    # corpus_lda = lda_model[corpus]
    #
    # for l, t in izip(corpus_lda, corpus):
    #   print l, "#", t
    # for l in corpus_lda:
    #     print(l)
    # for topic in lda_model.show_topics(num_topics=num_topics, num_words=50):
    #     print(topic)

    return lda_model


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
    cached_stop_words = set(stopwords.words("english"))
    body = []
    processed = []

    # remove common words and tokenize
    # texts = [[word for word in document.lower().split() if word not in stopwords.words('english')]
    #          for document in reviews]

    for i in range(0, len(document_list)):
        body.append(document_list[i].lower())

    for entry in body:
        row = tokenizer.tokenize(entry)
        processed.append([word for word in row if word not in cached_stop_words])

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


def get_user_item_contexts(records, lda_model, user_id, apply_filter=False):

    if apply_filter:
        user_records = ETLUtils.filter_records(records, 'user_id', [user_id])
    else:
        user_records = records

    if not user_records:
        return {}

    items_reviews = {}

    for record in user_records:
        review_text = record['text']
        context = get_topic_distribution(review_text, lda_model)
        items_reviews[record['offering_id']] = context

    return items_reviews


def get_topic_distribution(review_text, lda_model, minimum_probability,
                           text_sampling_proportion=None):
        """

        :type review_text: str
        :type lda_model: LdaModel
        :type minimum_probability: float
        :type text_sampling_proportion: float
        :param text_sampling_proportion: a float in the range [0,1] that
        indicates the proportion of text that should be sampled from the review
         text. If None then all the review text is taken
        """
        review_bow = create_bag_of_words([review_text])

        if text_sampling_proportion is not None and len(review_bow[0]) > 0:

            num_words = int(text_sampling_proportion * len(review_bow[0]))
            review_bow = [
                numpy.random.choice(review_bow[0], num_words, replace=False)
            ]

        dictionary = corpora.Dictionary(review_bow)
        corpus = dictionary.doc2bow(review_bow[0])
        lda_corpus = lda_model.get_document_topics(
            corpus, minimum_probability=minimum_probability)

        topic_distribution = numpy.zeros(lda_model.num_topics)
        for pair in lda_corpus:
            topic_distribution[pair[0]] = pair[1]

        return topic_distribution


my_documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"
]

# print(create_bag_of_words(my_documents))

# my_bag_of_words = create_bag_of_words(my_documents)
# my_topic_model = discover_topics(my_documents, 3)
# my_dictionary = corpora.Dictionary(my_bag_of_words)
# my_dictionary.filter_extremes(2, 0.6)
# my_corpus = [my_dictionary.doc2bow(text) for text in my_bag_of_words]
# print(my_topic_model[my_corpus])
# for t in my_topic_model[my_corpus]:
#     print(t)
