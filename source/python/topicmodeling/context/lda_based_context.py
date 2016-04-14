import copy
import time

import operator
from gensim import corpora

from topicmodeling.context import lda_context_utils
from topicmodeling.context import context_utils
from utils.constants import Constants

__author__ = 'fpena'


class LdaBasedContext:

    def __init__(self, records):
        self.records = records
        self.alpha = Constants.LDA_ALPHA
        self.beta = Constants.LDA_BETA
        self.epsilon = Constants.LDA_EPSILON
        self.specific_reviews = None
        self.generic_reviews = None
        self.all_nouns = None
        self.all_senses = None
        self.sense_groups = None
        self.review_topics_list = None
        self.num_topics = Constants.LDA_NUM_TOPICS
        self.topics = range(self.num_topics)
        self.topic_model = None
        self.context_rich_topics = None
        self.topic_ratio_map = None
        self.specific_dictionary = None
        self.specific_corpus = None
        self.generic_dictionary = None
        self.generic_corpus = None
        self.max_words = None

    def separate_reviews(self):

        self.specific_reviews = []
        self.generic_reviews = []

        for record in self.records:
            if record[Constants.PREDICTED_CLASS_FIELD] == 'specific':
                self.specific_reviews.append(record)
            if record[Constants.PREDICTED_CLASS_FIELD] == 'generic':
                self.generic_reviews.append(record)

    def generate_review_corpus(self):

        self.separate_reviews()

        specific_reviews_text = \
            context_utils.get_text_from_reviews(self.specific_reviews)
        generic_reviews_text = \
            context_utils.get_text_from_reviews(self.generic_reviews)

        specific_bow = lda_context_utils.create_bag_of_words(
            specific_reviews_text)
        generic_bow = \
            lda_context_utils.create_bag_of_words(generic_reviews_text)

        self.specific_dictionary = corpora.Dictionary(specific_bow)
        self.specific_dictionary.filter_extremes()
        self.specific_corpus = \
            [self.specific_dictionary.doc2bow(text) for text in specific_bow]

        self.generic_dictionary = corpora.Dictionary(generic_bow)
        self.generic_dictionary.filter_extremes()
        self.generic_corpus = \
            [self.generic_dictionary.doc2bow(text) for text in generic_bow]

    def build_topic_model(self):
        print('%s: building topic model' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))
        self.topic_model = lda_context_utils.build_topic_model_from_corpus(
            self.specific_corpus, self.specific_dictionary)
        print('%s: topic model built' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

    def update_reviews_with_topics(self):
        lda_context_utils.update_reviews_with_topics(
            self.topic_model, self.specific_corpus, self.specific_reviews,
            self.epsilon)
        lda_context_utils.update_reviews_with_topics(
            self.topic_model, self.generic_corpus, self.generic_reviews,
            self.epsilon)

        print('%s: updated reviews with topics' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

    def get_context_rich_topics(self):
        """
        Returns a list with the topics that are context rich and their
        specific/generic frequency ratio

        :rtype: list[(int, float)]
        :return: a list of pairs where the first position of the pair indicates
        the topic and the second position indicates the specific/generic
        frequency ratio
        """

        # numpy.random.seed(0)
        topic_ratio_map = {}
        non_contextual_topics = set()
        for topic in range(self.num_topics):
            weighted_frq = lda_context_utils.calculate_topic_weighted_frequency(
                topic, self.records)
            specific_weighted_frq = \
                lda_context_utils.calculate_topic_weighted_frequency(
                    topic, self.specific_reviews)
            generic_weighted_frq = \
                lda_context_utils.calculate_topic_weighted_frequency(
                    topic, self.generic_reviews)

            if weighted_frq < self.alpha:
                non_contextual_topics.add(topic)

            if generic_weighted_frq == 0:
                ratio = 'N/A'
            else:
                ratio = specific_weighted_frq / generic_weighted_frq

            # print('topic: %d --> ratio: %f\tspecific: %f\tgeneric: %f' %
            #       (topic, ratio, specific_weighted_frq, generic_weighted_frq))

            if ratio < self.beta:
                non_contextual_topics.add(topic)

            topic_ratio_map[topic] = ratio

        self.topic_ratio_map = copy.deepcopy(topic_ratio_map)

        # lda_context_utils.export_topics(self.topic_model, topic_ratio_map)
        # print('%s: exported topics' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        for topic in non_contextual_topics:
            topic_ratio_map.pop(topic)

        # print('non contextual topics', len(non_contextual_topics))
        # for topic in topic_ratio_map.keys():
        #     print(topic, topic_ratio_map[topic])
        #
        sorted_topics = sorted(
            topic_ratio_map.items(), key=operator.itemgetter(1), reverse=True)

        # for topic in sorted_topics:
        #     topic_index = topic[0]
        #     ratio = topic[1]
        #     print('topic', ratio, topic_index, self.topic_model.print_topic(topic_index, topn=50))

        # print('num_topics', len(self.topics))
        print('context topics: %d' % len(topic_ratio_map))
        self.context_rich_topics = sorted_topics
        # print(self.context_rich_topics)
        self.max_words = []
        for topic in self.context_rich_topics:
            self.max_words.append(
                self.topic_model.show_topic(topic[0], 1)[0][1])

        return sorted_topics

    def get_all_topics(self):
        """
        Returns a list with all the topics after training the LDA model with all
        the reviews (specific + generic)

        :rtype: list[(int, float)]
        :return: a list of pairs where the first position of the pair indicates
        the topic and the second position has a 1.0 value (this is just to have
        the results with the same format as the get_context_rich_topics()
        method)
        """

        reviews_text =\
            context_utils.get_text_from_reviews(self.records)

        bag_of_words = lda_context_utils.create_bag_of_words(reviews_text)

        dictionary = corpora.Dictionary(bag_of_words)
        dictionary.filter_extremes()
        corpus =\
            [dictionary.doc2bow(text) for text in bag_of_words]

        self.topic_model =\
            lda_context_utils.build_topic_model_from_corpus(corpus, dictionary)

        lda_context_utils.update_reviews_with_topics(
            self.topic_model, corpus, self.records, self.epsilon)

        topic_ratio_map = {}

        for topic in range(self.num_topics):
            topic_ratio_map[topic] = 1

        # export_all_topics(self.topic_model)
        # print('%s: exported topics' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        sorted_topics = sorted(
            topic_ratio_map.items(), key=operator.itemgetter(1), reverse=True)

        self.context_rich_topics = sorted_topics

        return sorted_topics

    def find_contextual_topics(self, records, text_sampling_proportion=None):
        for record in records:
            # numpy.random.seed(0)
            topic_distribution = lda_context_utils.get_topic_distribution(
                record[Constants.TEXT_FIELD], self.topic_model, self.epsilon,
                text_sampling_proportion, self.max_words
            )
            record[Constants.TOPICS_FIELD] = topic_distribution

            topics_map = {}
            for i in self.context_rich_topics:
                topic_id = 'topic' + str(i[0])
                topics_map[topic_id] = topic_distribution[i[0]]

            record[Constants.CONTEXT_TOPICS_FIELD] = topics_map

        # print(self.context_rich_topics)
        # print('total_topics', len(self.context_rich_topics))

        return records


def main():
    pass

# numpy.random.seed(0)
#
# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
