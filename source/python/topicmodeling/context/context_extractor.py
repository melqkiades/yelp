import copy
import time

import operator

import math

from etl import ETLUtils
from utils.constants import Constants


class ContextExtractor:

    def __init__(self, records):
        self.records = records
        self.target_reviews = None
        self.non_target_reviews = None
        self.num_topics = Constants.TOPIC_MODEL_NUM_TOPICS
        self.context_rich_topics = None
        self.topic_weighted_frequency_map = None
        self.topic_ratio_map = None
        self.target_bows = None
        self.non_target_bows = None
        self.lda_beta_comparison_operator = None
        if Constants.LDA_BETA_COMPARISON_OPERATOR == 'gt':
            self.lda_beta_comparison_operator = operator.gt
        elif Constants.LDA_BETA_COMPARISON_OPERATOR == 'lt':
            self.lda_beta_comparison_operator = operator.lt
        elif Constants.LDA_BETA_COMPARISON_OPERATOR == 'ge':
            self.lda_beta_comparison_operator = operator.ge
        elif Constants.LDA_BETA_COMPARISON_OPERATOR == 'le':
            self.lda_beta_comparison_operator = operator.le
        elif Constants.LDA_BETA_COMPARISON_OPERATOR == 'eq':
            self.lda_beta_comparison_operator = operator.le
        else:
            raise ValueError('Comparison operator not supported for LDA beta')

    def separate_reviews(self):

        self.target_reviews = []
        self.non_target_reviews = []

        for record in self.records:
            if record[Constants.TOPIC_MODEL_TARGET_FIELD] == \
                    Constants.TOPIC_MODEL_TARGET_REVIEWS:
                self.target_reviews.append(record)
            else:
                self.non_target_reviews.append(record)

        print("num target reviews: %d" % len(self.target_reviews))
        print("num non-target reviews: %d" % len(self.non_target_reviews))

    def generate_review_bows(self):

        self.separate_reviews()

        self.target_bows = []
        for record in self.target_reviews:
            self.target_bows.append(" ".join(record[Constants.BOW_FIELD]))
        self.non_target_bows = []
        for record in self.non_target_reviews:
            self.non_target_bows.append(" ".join(record[Constants.BOW_FIELD]))

    def get_context_rich_topics(self):
        """
        Returns a list with the topics that are context rich and their
        specific/generic frequency ratio

        :rtype: list[(int, float)]
        :return: a list of pairs where the first position of the pair indicates
        the topic and the second position indicates the specific/generic
        frequency ratio
        """
        if Constants.TOPIC_WEIGHTING_METHOD == Constants.ALL_TOPICS or Constants.TOPIC_MODEL_TARGET_REVIEWS is None:
            self.topic_ratio_map = {}
            self.topic_weighted_frequency_map = {}

            for topic in range(self.num_topics):
                self.topic_ratio_map[topic] = 1
                self.topic_weighted_frequency_map[topic] = 1

            sorted_topics = sorted(
                self.topic_ratio_map.items(), key=operator.itemgetter(1),
                reverse=True)

            self.context_rich_topics = sorted_topics
            print('all_topics')
            print('context topics: %d' % len(self.context_rich_topics))
            return sorted_topics

        topic_ratio_map = {}
        self.topic_weighted_frequency_map = {}
        lower_than_alpha_count = 0.0
        lower_than_beta_count = 0.0
        non_contextual_topics = set()
        for topic in range(self.num_topics):
            # print('topic: %d' % topic)
            weighted_frq = self.calculate_topic_weighted_frequency(
                topic, self.records)
            target_weighted_frq = \
                self.calculate_topic_weighted_frequency(
                    topic, self.target_reviews)
            non_target_weighted_frq = \
                self.calculate_topic_weighted_frequency(
                    topic, self.non_target_reviews)

            if weighted_frq < Constants.CONTEXT_EXTRACTOR_ALPHA:
                non_contextual_topics.add(topic)
                # print('non-contextual_topic: %d' % topic)
                lower_than_alpha_count += 1.0

            if non_target_weighted_frq == 0:
                # We can't know if the topic is good or not
                non_contextual_topics.add(topic)
                ratio = 'N/A'
                # non_contextual_topics.add(topic)
            else:
                ratio = target_weighted_frq / non_target_weighted_frq
                print('topic: %d, specific_frq: %f, generic_frq: %f' % (topic, target_weighted_frq, non_target_weighted_frq))

            # print('topic: %d --> ratio: %f\tspecific: %f\tgeneric: %f' %
            #       (topic, ratio, target_weighted_frq, non_target_weighted_frq))

            if self.lda_beta_comparison_operator(
                    ratio, Constants.CONTEXT_EXTRACTOR_BETA):
                non_contextual_topics.add(topic)
                lower_than_beta_count += 1.0

            topic_ratio_map[topic] = ratio
            self.topic_weighted_frequency_map[topic] = weighted_frq

        self.topic_ratio_map = copy.deepcopy(topic_ratio_map)

        for topic in non_contextual_topics:
            topic_ratio_map.pop(topic)

        sorted_topics = sorted(
            topic_ratio_map.items(), key=operator.itemgetter(1), reverse=True)

        print('context topics: %d' % len(topic_ratio_map))
        print('topics lower than alpha: %d' % lower_than_alpha_count)
        print('topics lower than beta: %d' % lower_than_beta_count)
        self.context_rich_topics = sorted_topics
        print(self.context_rich_topics)

        return sorted_topics

    @staticmethod
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
                        if math.isnan(review_topic[1]):
                            print(topic, review)
                    else:
                        raise ValueError(
                            'Topic weighting method not recognized')

        print('num_reviews: %f, len(reviews): %f' % (num_reviews, len(reviews)))

        return num_reviews / len(reviews)

    def clear_reviews(self):
        self.records = None
        self.target_reviews = None
        self.non_target_reviews = None
        self.target_bows = None
        self.non_target_bows = None

    def find_contextual_topics(self, records, text_sampling_proportion=None):
        for record in records:
            # numpy.random.seed(0)
            topic_distribution = record[Constants.TOPICS_FIELD]

            topics_map = {}
            context_topics_sum = 0.0
            for i in self.context_rich_topics:
                topic_index = i[0]
                topic_weight = topic_distribution[topic_index][1]
                topic_id = 'topic_%02d' % topic_index
                topics_map[topic_id] = topic_weight
                context_topics_sum += topic_weight

            topics_map['nocontexttopics'] = 1 - context_topics_sum

            record[Constants.CONTEXT_TOPICS_FIELD] = topics_map

        return records


def main():

    # records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)
    records = ETLUtils.load_json_file(Constants.RECSYS_TOPICS_PROCESSED_RECORDS_FILE)

    print('num_reviews', len(records))
    # lda_context_utils.discover_topics(my_reviews, 150)
    context_extractor = ContextExtractor(records)
    context_extractor.separate_reviews()
    context_extractor.get_context_rich_topics()

# numpy.random.seed(0)

# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
