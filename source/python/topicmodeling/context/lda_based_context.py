from gensim.models import ldamodel
import numpy

import time
import operator
from gensim import corpora
import cPickle as pickle
from topicmodeling.context import lda_context_utils
from topicmodeling.context import context_utils
from topicmodeling.context import reviews_clusterer
from topicmodeling.context.review import Review


__author__ = 'fpena'


class LdaBasedContext:

    def __init__(self, reviews=None, text_reviews=None):
        self.text_reviews = text_reviews
        self.alpha = 0.005
        self.beta = 1.0
        self.epsilon = 0.01
        self.reviews = reviews
        self.specific_reviews = None
        self.generic_reviews = None
        self.all_nouns = None
        self.all_senses = None
        self.sense_groups = None
        self.review_topics_list = None
        self.num_topics = 150
        self.topics = range(self.num_topics)
        self.topic_model = None

    def init_reviews(self):
        if not self.reviews:
            self.init_from_text(self.text_reviews)

    def init_from_text(self, text_reviews):
        self.text_reviews = text_reviews
        self.reviews = []

        index = 0
        print('num_reviews', len(self.text_reviews))
        for text_review in self.text_reviews:
            self.reviews.append(Review(text_review))
            print('index', index)
            index += 1

    def separate_reviews(self):
        """
        Separates the reviews into specific and generic. The separation is done
        by clustering

        """
        print('separating reviews', time.strftime("%H:%M:%S"))

        cluster_labels = reviews_clusterer.cluster_reviews(self.reviews)
        review_clusters =\
            reviews_clusterer.split_list_by_labels(self.reviews, cluster_labels)

        self.specific_reviews = review_clusters[0]
        self.generic_reviews = review_clusters[1]

        print('finished separating reviews', time.strftime("%H:%M:%S"))

    def get_context_rich_topics(self):
        """
        Returns a list with the topics that are context rich and their
        specific/generic frequency ratio

        :rtype: list[(int, float)]
        :return: a list of pairs where the first position of the pair indicates
        the topic and the second position indicates the specific/generic
        frequency ratio
        """
        self.separate_reviews()

        specific_reviews_text =\
            context_utils.get_text_from_reviews(self.specific_reviews)
        generic_reviews_text =\
            context_utils.get_text_from_reviews(self.generic_reviews)

        specific_bow =\
            lda_context_utils.create_bag_of_words(specific_reviews_text)
        generic_bow =\
            lda_context_utils.create_bag_of_words(generic_reviews_text)

        specific_dictionary = corpora.Dictionary(specific_bow)
        specific_dictionary.filter_extremes(2, 0.6)
        specific_corpus =\
            [specific_dictionary.doc2bow(text) for text in specific_bow]

        generic_dictionary = corpora.Dictionary(generic_bow)
        generic_dictionary.filter_extremes(2, 0.6)
        generic_corpus =\
            [generic_dictionary.doc2bow(text) for text in generic_bow]

        self.topic_model = ldamodel.LdaModel(
            specific_corpus, id2word=specific_dictionary,
            num_topics=self.num_topics, minimum_probability=self.epsilon)

        lda_context_utils.update_reviews_with_topics(
            self.topic_model, specific_corpus, self.specific_reviews)
        lda_context_utils.update_reviews_with_topics(
            self.topic_model, generic_corpus, self.generic_reviews)

        topic_ratio_map = {}
        ratio_topics = 0

        for topic in range(self.num_topics):
            weighted_frq = lda_context_utils.calculate_topic_weighted_frequency(
                topic, self.reviews)
            specific_weighted_frq = \
                lda_context_utils.calculate_topic_weighted_frequency(
                    topic, self.specific_reviews)
            generic_weighted_frq = \
                lda_context_utils.calculate_topic_weighted_frequency(
                    topic, self.generic_reviews)

            if weighted_frq < self.alpha:
                continue

            ratio = specific_weighted_frq / generic_weighted_frq

            if ratio < self.beta:
                continue

            ratio_topics += 1
            topic_ratio_map[topic] = ratio

        sorted_topics = sorted(
            topic_ratio_map.items(), key=operator.itemgetter(1), reverse=True)

        print('num_topics', len(self.topics))
        print('ratio_topics', ratio_topics)

        return sorted_topics


def main():
    # reviews_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review_hotels.json"
    # my_reviews = context_utils.load_reviews(reviews_file)
    # print("reviews:", len(my_reviews))
    #
    # my_reviews = None
    my_file = '/Users/fpena/tmp/sentences_hotel.pkl'
    # with open(my_file, 'wb') as write_file:
    #     pickle.dump(self.reviews, write_file, pickle.HIGHEST_PROTOCOL)

    with open(my_file, 'rb') as read_file:
        my_reviews = pickle.load(read_file)

    # lda_context_utils.discover_topics(my_reviews, 150)
    lda_based_context = LdaBasedContext(reviews=my_reviews)
    lda_based_context.init_reviews()
    # my_topics = lda_based_context.get_context_rich_topics()
    my_topics = lda_based_context.get_context_rich_topics()
    print(my_topics)
    print('total_topics', len(my_topics))

# numpy.random.seed(0)
#
# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
