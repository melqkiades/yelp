import copy
import time

import operator

import numpy
from sklearn import decomposition

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from etl import ETLUtils
from utils.constants import Constants


class NmfContextExtractor:

    def __init__(self, records):
        self.records = records
        self.target_reviews = None
        self.non_target_reviews = None
        self.num_topics = Constants.TOPIC_MODEL_NUM_TOPICS
        self.topics = range(self.num_topics)
        self.topic_model = None
        self.tfidf_vectorizer = None
        self.context_rich_topics = None
        self.topic_weighted_frequency_map = None
        self.topic_ratio_map = None
        self.target_bows = None
        self.non_target_bows = None
        self.document_term_matrix = None
        self.document_topic_matrix = None
        self.topic_term_matrix = None
        self.terms = None
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

    def build_document_term_matrix(self):

        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=ENGLISH_STOP_WORDS, lowercase=True,
            strip_accents="unicode",
            use_idf=True, norm="l2", min_df=Constants.MIN_DICTIONARY_WORD_COUNT,
            max_df=Constants.MAX_DICTIONARY_WORD_COUNT, ngram_range=(1, 1))
        self.document_term_matrix = \
            self.tfidf_vectorizer.fit_transform(self.target_bows)

        vocabulary = self.tfidf_vectorizer.vocabulary_
        num_terms = len(vocabulary)
        self.terms = [""] * num_terms
        for term in vocabulary.keys():
            self.terms[vocabulary[term]] = term

        print "Created document-term matrix of size %d x %d" % (
            self.document_term_matrix.shape[0],
            self.document_term_matrix.shape[1]
        )

    def build_topic_model(self):
        print('%s: building NMF topic model' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        self.topic_model = decomposition.NMF(
            init="nndsvd", n_components=self.num_topics,
            max_iter=Constants.TOPIC_MODEL_ITERATIONS)
        self.document_topic_matrix =\
            self.topic_model.fit_transform(self.document_term_matrix)
        self.topic_term_matrix = self.topic_model.components_

        print('%s: topic model built' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

    def build_single_topic_model(self):
        # print('%s: building NMF topic model' %
        #       time.strftime("%Y/%m/%d-%H:%M:%S"))

        topic_model = decomposition.NMF(
            init="nndsvd", n_components=self.num_topics,
            max_iter=Constants.TOPIC_MODEL_ITERATIONS,
            alpha=Constants.NMF_REGULARIZATION,
            l1_ratio=Constants.NMF_REGULARIZATION_RATIO
        )
        topic_model.fit_transform(self.document_term_matrix)
        topic_term_matrix = topic_model.components_

        return topic_term_matrix

    def build_stable_topic_model(self):

        matrices = []
        for i in range(Constants.TOPIC_MODEL_PASSES):
            topic_term_matrix = self.build_single_topic_model().transpose()
            matrices.append(topic_term_matrix)

        stack_matrix = numpy.hstack(matrices)
        stack_matrix = normalize(stack_matrix, axis=0)
        stack_matrix = stack_matrix.transpose()

        print "Stack matrix M of size %s" % str(stack_matrix.shape)

        self.topic_model = decomposition.NMF(
            init="nndsvd", n_components=self.num_topics,
            max_iter=Constants.TOPIC_MODEL_ITERATIONS,
            alpha=Constants.NMF_REGULARIZATION,
            l1_ratio=Constants.NMF_REGULARIZATION_RATIO
        )

        self.document_topic_matrix = \
            self.topic_model.fit_transform(stack_matrix)
        self.topic_term_matrix = self.topic_model.components_

        row_sums = self.topic_term_matrix.sum(axis=1)
        self.topic_term_matrix /= row_sums[:, numpy.newaxis]

        print "Generated factor W of size %s and factor H of size %s" % (
            str(self.document_topic_matrix.shape),
            str(self.topic_term_matrix.shape)
        )

        # return model

    def update_reviews_with_topics(self):

        self.update_documents_with_topics(self.target_reviews, self.target_bows)
        self.update_documents_with_topics(
            self.non_target_reviews, self.non_target_bows)

        print('%s: updated reviews with topics' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

    def update_documents_with_topics(self, reviews_list, bow_list):
        document_term_matrix = \
            self.tfidf_vectorizer.transform(bow_list)
        document_topic_matrix = \
            self.topic_model.transform(document_term_matrix)
        for review_index in range(len(reviews_list)):
            review = reviews_list[review_index]
            review[Constants.TOPICS_FIELD] = \
                [(i, document_topic_matrix[review_index][i])
                 for i in range(self.num_topics)]

    def get_context_rich_topics(self):
        """
        Returns a list with the topics that are context rich and their
        specific/generic frequency ratio

        :rtype: list[(int, float)]
        :return: a list of pairs where the first position of the pair indicates
        the topic and the second position indicates the specific/generic
        frequency ratio
        """
        if Constants.TOPIC_WEIGHTING_METHOD == Constants.ALL_TOPICS:
            self.topic_ratio_map = {}
            self.topic_weighted_frequency_map = {}

            for topic in range(self.num_topics):
                self.topic_ratio_map[topic] = 1
                self.topic_weighted_frequency_map[topic] = 1

            # export_all_topics(self.topic_model)
            # print('%s: exported topics' % time.strftime("%Y/%m/%d-%H:%M:%S"))

            sorted_topics = sorted(
                self.topic_ratio_map.items(), key=operator.itemgetter(1),
                reverse=True)

            self.context_rich_topics = sorted_topics
            print('all_topics')
            print('context topics: %d' % len(self.context_rich_topics))
            return sorted_topics

        # numpy.random.seed(0)
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

            # print('topic: %d --> ratio: %f\tspecific: %f\tgeneric: %f' %
            #       (topic, ratio, target_weighted_frq, non_target_weighted_frq))

            if self.lda_beta_comparison_operator(
                    ratio, Constants.CONTEXT_EXTRACTOR_BETA):
                non_contextual_topics.add(topic)
                lower_than_beta_count += 1.0
                # print('non-contextual_topic: %d' % topic)

            topic_ratio_map[topic] = ratio
            self.topic_weighted_frequency_map[topic] = weighted_frq

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
                    else:
                        raise ValueError(
                            'Topic weighting method not recognized')

        return num_reviews / len(reviews)

    def find_contextual_topics(self, records, text_sampling_proportion=None):
        for record in records:
            # numpy.random.seed(0)
            topic_distribution =\
                self.get_topic_distribution(record)

            # We calculate the sum of the probabilities of the contextual topics
            # to then normalize the contextual vector
            context_topics_sum = 0.0
            # print('context rich topics', self.context_rich_topics)
            for i in self.context_rich_topics:
                context_topics_sum += topic_distribution[i[0]]

            topics_map = {}
            for i in self.context_rich_topics:
                topic_id = 'topic' + str(i[0])
                if context_topics_sum > 0:
                    topics_map[topic_id] = \
                        topic_distribution[i[0]] / 1.0
                else:
                    topics_map[topic_id] = 0.0

            topics_map['nocontexttopics'] = 1 - context_topics_sum

            record[Constants.CONTEXT_TOPICS_FIELD] = topics_map

        # print(self.context_rich_topics)
        # print('total_topics', len(self.context_rich_topics))

        return records

    def print_topic(self, topic_index, num_terms=10):
        top_indices = numpy.argsort(
            self.topic_term_matrix[topic_index, :])[::-1][0:num_terms]
        term_ranking = [
            '%.3f*%s' % (self.topic_term_matrix[topic_index][i], self.terms[i])
            for i in top_indices
        ]
        # terms = ", ".join(term_ranking)
        # string = "Topic %d: %s" % (topic_index, ", ".join(term_ranking))
        # term_ranking_list.append(string)
        # term_ranking_list.append(term_ranking)
        # print(top_indices)
        # term_probability = [H[topic_index][i] for i in top_indices]
        # print(sum(term_probability))
        topic_string = " + ".join(term_ranking)
        # print("Topic %d: %s" % (topic_index, topic_string))
        return topic_string

    def print_topic_model(self, num_terms=10):

        return [
            self.print_topic(topic_id, num_terms)
            for topic_id in range(self.num_topics)
        ]

    def get_topic_distribution(self, record):
        corpus = " ".join(record[Constants.BOW_FIELD])
        document_term_matrix = \
            self.tfidf_vectorizer.transform([corpus])
        document_topic_matrix = self.topic_model.transform(document_term_matrix)

        return document_topic_matrix[0]

    def clear_reviews(self):
        self.records = None
        self.target_reviews = None
        self.non_target_reviews = None
        self.target_bows = None
        self.non_target_bows = None
        self.document_term_matrix = None
        self.document_topic_matrix = None


def main():

    records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)

    print('num_reviews', len(records))
    # lda_context_utils.discover_topics(my_reviews, 150)
    context_extractor = NmfContextExtractor(records)
    context_extractor.generate_review_bows()
    context_extractor.build_document_term_matrix()
    # context_extractor.build_topic_model()
    context_extractor.build_stable_topic_model()
    context_extractor.print_topic_model()
    context_extractor.update_reviews_with_topics()
    context_extractor.get_context_rich_topics()

# numpy.random.seed(0)

# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
