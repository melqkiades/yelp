from gensim.models import ldamodel

import time
import operator
from gensim import corpora
import cPickle as pickle
import numpy
from etl import ETLUtils
from topicmodeling.context import lda_context_utils
from topicmodeling.context import context_utils
from topicmodeling.context import reviews_clusterer
from topicmodeling.context.review import Review
from topicmodeling.context.reviews_classifier import ReviewsClassifier
from topicmodeling.context import review_metrics_extractor

__author__ = 'fpena'


class LdaBasedContext:

    def __init__(self, records):
        self.records = records
        # self.reviews = reviews
        self.alpha = 0.005
        self.beta = 1.0
        self.epsilon = 0.01
        self.specific_reviews = None
        self.generic_reviews = None
        self.all_nouns = None
        self.all_senses = None
        self.sense_groups = None
        self.review_topics_list = None
        self.num_topics = 150
        self.topics = range(self.num_topics)
        self.topic_model = None
        # self.training_set_file = None
        # self.training_reviews_file = None
        self.context_rich_topics = None

        # self.init_reviews()

    # def init_reviews(self):
    #     if not self.reviews:
    #         self.init_from_records(self.records)
    #
    # def init_from_records(self, records):
    #     self.reviews = []
    #
    #     index = 0
    #     print('num_reviews', len(self.records))
    #     for record in self.records:
    #         self.reviews.append(Review(record['text']))
    #         print('index', index)
    #         index += 1

    def separate_reviews(self):

        self.specific_reviews = []
        self.generic_reviews = []

        for record in self.records:
            if record['type'] == 'specific':
                self.specific_reviews.append(record)
            if record['type'] == 'generic':
                self.generic_reviews.append(record)

    def get_context_rich_topics(self):
        """
        Returns a list with the topics that are context rich and their
        specific/generic frequency ratio

        :rtype: list[(int, float)]
        :return: a list of pairs where the first position of the pair indicates
        the topic and the second position indicates the specific/generic
        frequency ratio
        """
        # self.separate_reviews()
        # self.separate_reviews_by_classifier()
        self.separate_reviews()

        specific_reviews_text =\
            context_utils.get_text_from_reviews(self.specific_reviews)
        generic_reviews_text =\
            context_utils.get_text_from_reviews(self.generic_reviews)

        specific_bow = lda_context_utils.create_bag_of_words(
            specific_reviews_text)
        generic_bow =\
            lda_context_utils.create_bag_of_words(generic_reviews_text)

        specific_dictionary = corpora.Dictionary(specific_bow)
        specific_dictionary.filter_extremes()
        specific_corpus =\
            [specific_dictionary.doc2bow(text) for text in specific_bow]

        generic_dictionary = corpora.Dictionary(generic_bow)
        generic_dictionary.filter_extremes()
        generic_corpus =\
            [generic_dictionary.doc2bow(text) for text in generic_bow]

        self.topic_model = ldamodel.LdaModel(
            specific_corpus, id2word=specific_dictionary,
            # num_topics=self.num_topics, minimum_probability=self.epsilon)
            num_topics=self.num_topics, minimum_probability=self.epsilon,
            passes=10, iterations=500)
        # print('super trained')

        lda_context_utils.update_reviews_with_topics(
            self.topic_model, specific_corpus, self.specific_reviews)
        lda_context_utils.update_reviews_with_topics(
            self.topic_model, generic_corpus, self.generic_reviews)

        topic_ratio_map = {}
        ratio_topics = 0

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
                continue

            # print('specific_weighted_frq', specific_weighted_frq)
            # print('generic_weighted_frq', generic_weighted_frq)

            ratio = (specific_weighted_frq + 1) / (generic_weighted_frq + 1)

            if ratio < self.beta:
                continue

            ratio_topics += 1
            topic_ratio_map[topic] = ratio

        sorted_topics = sorted(
            topic_ratio_map.items(), key=operator.itemgetter(1), reverse=True)

        # for topic in sorted_topics:
        #     topic_index = topic[0]
        #     ratio = topic[1]
        #     print('topic', ratio, topic_index, self.topic_model.print_topic(topic_index, topn=50))

        # print('num_topics', len(self.topics))
        # print('ratio_topics', ratio_topics)
        self.context_rich_topics = sorted_topics

        return sorted_topics

    def find_contextual_topics(self, records):

        print('lda num_reviews', len(records))
        # print(self.context_rich_topics)
        # print('total_topics', len(self.context_rich_topics))
        headers = ['stars', 'user_id', 'business_id']
        for i in self.context_rich_topics:
            topic_id = 'topic' + str(i[0])
            headers.append(topic_id)

        # output_records = []

        for record in records:
            topic_distribution =\
                lda_context_utils.get_topic_distribution(
                    record['text'], self.topic_model)

            # output_record = {}
            # record['user_id'] = record['user_id']
            # output_record['item_id'] = record['business_id']
            # output_record['rating'] = record['stars']

            for i in self.context_rich_topics:
                topic_id = 'topic' + str(i[0])
                record[topic_id] = topic_distribution[i[0]]

            # print(output_record)
            # output_records.append(output_record)

        print(self.context_rich_topics)
        print('total_topics', len(self.context_rich_topics))
        # print(headers)

        return records

    def export_contextual_records(
            self, records_file, binary_reviews_file=None, json_file=None,
            csv_file=None):

        # reviews_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_restaurants_shuffled.json"
        # binary_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/reviews_restaurant_shuffled.pkl'
        records = ETLUtils.load_json_file(records_file)
        # reviews = None
        #
        # if binary_reviews_file is not None:
        #     with open(binary_reviews_file, 'rb') as read_file:
        #         reviews = pickle.load(read_file)

        output_records, headers = self.find_contextual_topics(records)

        # json_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_restaurant_context_shuffled.json"
        # csv_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_restaurant_context_shuffled.csv"
        if json_file is not None:
            ETLUtils.save_json_file(json_file, output_records)
        if csv_file is not None:
            ETLUtils.save_csv_file(csv_file, output_records, headers)


def test_reviews_classfier():
    records_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_hotel_shuffled.json"
    binary_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/reviews_hotel_shuffled.pkl'
    training_records_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_hotel_reviews.json'
    training_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_hotel_reviews.pkl'

    records = ETLUtils.load_json_file(records_file)

    with open(binary_reviews_file, 'rb') as read_file:
        my_binary_reviews = pickle.load(read_file)

    lda_based_context = LdaBasedContext(records)
    lda_based_context.training_set_file = training_records_file
    lda_based_context.training_reviews_file = training_reviews_file
    lda_based_context.reviews = my_binary_reviews
    lda_based_context.init_reviews()
    # lda_based_context.separate_reviews_by_classifier()

    print('\n\n\n\n\nSpecific reviews')
    for review in lda_based_context.specific_reviews[:10]:
        print(review.text, '***')

    print('\n\n\nGeneric reviews')
    for review in lda_based_context.generic_reviews[:10]:
        print(review.text, '***')


def main():

    ITEM_TYPE = 'hotel'
    # dataset = 'restaurant'
    DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/'
    RECORDS_FILE = DATASET_FOLDER + 'reviews_' + ITEM_TYPE + '_shuffled.json'
    print(RECORDS_FILE)

    # my_training_records_file = '/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'
    my_training_records = ETLUtils.load_json_file(RECORDS_FILE)
    # my_training_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/reviews_' + dataset + '_shuffled.pkl'
    # my_reviews = context_utils.load_reviews(reviews_file)
    # print("reviews:", len(my_reviews))
    #
    # my_reviews = None
    # my_file = '/Users/fpena/tmp/reviews_restaurant_shuffled.pkl'
    # my_file = '/Users/fpena/tmp/sentences_hotel.pkl'
    # my_file = '/Users/fpena/tmp/reviews_hotel.pkl'
    # my_file = '/Users/fpena/tmp/reviews_spa.pkl'

    # with open(my_file, 'wb') as write_file:
    #     pickle.dump(self.reviews, write_file, pickle.HIGHEST_PROTOCOL)
    # training_records_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_' + dataset + '_reviews.json'
    # training_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_' + dataset + '_reviews.pkl'

    # with open(my_training_reviews_file, 'rb') as read_file:
    #     my_training_reviews = pickle.load(read_file)

    # print('lda num_reviews', len(my_training_reviews))
    # lda_context_utils.discover_topics(my_reviews, 150)
    lda_based_context = LdaBasedContext(my_training_records)
    # lda_based_context.training_set_file = training_records_file
    # lda_based_context.training_reviews_file = training_reviews_file
    # lda_based_context.init_reviews()
    # lda_based_context.separate_reviews()

    # my_specific_reviews = []
    # for review in lda_based_context.specific_reviews:
    #     my_specific_reviews.append({'review': '*\t' + review.text.encode('utf-8')})
    #
    # ETLUtils.save_csv_file('/Users/fpena/tmp/specific_reviews.csv', my_specific_reviews, ['review'], '|')
    #
    # my_generic_reviews = []
    # for review in lda_based_context.generic_reviews:
    #     my_generic_reviews.append({'review': '*\t' + review.text.encode('utf-8')})
    #
    # ETLUtils.save_csv_file('/Users/fpena/tmp/generic_reviews.csv', my_generic_reviews, ['review'], '|')

    my_topics = lda_based_context.get_context_rich_topics()
    print(my_topics)
    print('total_topics', len(my_topics))

    # my_records_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_' + dataset + '_reviews.json'
    # my_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_' + dataset + '_reviews.pkl'
    # json_file = '/Users/fpena/UCC/Thesis/datasets/context/yelp_' + dataset + '_context_shuffled4.json'
    # csv_file = '/Users/fpena/UCC/Thesis/datasets/context/yelp_' + dataset + '_context_shuffled4.csv'
    # lda_based_context.export_contextual_records(my_records_file, my_reviews_file, json_file, csv_file)

    # ETLUtils.save_json_file(json_file, output_records)
    # ETLUtils.save_csv_file(csv_file, output_records, headers)


def tmp_function():
    var1 = {'a': 1, 'b': 2, 'predicted_class': 'specific', 'topics': None}
    var2 = {'a': 3, 'b': 4, 'predicted_class': 'generic', 'topics': None}
    var3 = {'a': 5, 'b': 6, 'predicted_class': 'specific', 'topics': None}
    var4 = {'a': 7, 'b': 8, 'predicted_class': 'generic', 'topics': None}

    list1 = [
        var1,
        var2,
        var3,
        var4
    ]

    specific, generic = separate_reviews(list1)
    # var1['a'] = 8

    generic[0]['topics'] = 'ohhh'

    print('all')
    for element in list1:
        print(element)
    print('specific')
    for element in specific:
        print(element)
    print('generic')
    for element in generic:
        print(element)


def separate_reviews(records):

    specific_reviews = []
    generic_reviews = []

    for record in records:
        if record['predicted_class'] == 'specific':
            specific_reviews.append(record)
        if record['predicted_class'] == 'generic':
            generic_reviews.append(record)

    return specific_reviews, generic_reviews

# numpy.random.seed(0)
#
# start = time.time()
# main()
# tmp_function()
# # test_reviews_classfier()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)


