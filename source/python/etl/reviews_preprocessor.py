import os
import random

import time

import operator

import langdetect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import numpy
from gensim import corpora
from nltk import PerceptronTagger
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from etl import ETLUtils
from etl import sampler_factory
from etl.reviews_dataset_analyzer import ReviewsDatasetAnalyzer
from nlp import nlp_utils
from topicmodeling.context import lda_context_utils
from topicmodeling.context import topic_model_creator
from topicmodeling.context.context_extractor import ContextExtractor
from topicmodeling.context.context_transformer import ContextTransformer
from topicmodeling.context.reviews_classifier import ReviewsClassifier
from topicmodeling.nmf_topic_extractor import NmfTopicExtractor
from tripadvisor.fourcity import extractor
from utils import utilities
from utils.constants import Constants
from utils.utilities import all_context_words


class ReviewsPreprocessor:

    def __init__(self, use_cache=False):
        self.use_cache = use_cache
        self.records = None
        self.dictionary = None

        classifier = Constants.DOCUMENT_CLASSIFIER
        classifiers = {
            'logistic_regression': LogisticRegression(C=100),
            'svc': SVC(),
            'kneighbors': KNeighborsClassifier(n_neighbors=10),
            'decision_tree': DecisionTreeClassifier(),
            'nu_svc': NuSVC(),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        self.classifier = classifiers[classifier]
        self.resampler = sampler_factory.create_sampler(
            Constants.RESAMPLER, Constants.DOCUMENT_CLASSIFIER_SEED)
        classifiers = None

    def load_records(self):
        print('%s: load records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        self.records = ETLUtils.load_json_file(Constants.RECORDS_FILE)

    def shuffle_records(self):
        print('%s: shuffle records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        random.shuffle(self.records)
        self.records = self.records

    def transform_yelp_records(self):
        new_records = []

        for record in self.records:

            user_id = record['user_id']
            item_id = record['business_id']

            new_records.append(
                {
                    Constants.REVIEW_ID_FIELD: record['review_id'],
                    Constants.USER_ID_FIELD: user_id,
                    Constants.ITEM_ID_FIELD: item_id,
                    Constants.RATING_FIELD: record['stars'],
                    Constants.TEXT_FIELD: record['text'],
                    Constants.USER_ITEM_KEY_FIELD: '%s|%s' % (str(user_id), str(item_id)),
                }
            )

        self.records = new_records

    def transform_fourcity_records(self):
        new_records = []

        for record in self.records:

            user_id = record['author']['id']
            item_id = record['offering_id']

            new_records.append(
                {
                    Constants.REVIEW_ID_FIELD: record['id'],
                    Constants.USER_ID_FIELD: user_id,
                    Constants.ITEM_ID_FIELD: item_id,
                    Constants.RATING_FIELD: record['ratings']['overall'],
                    Constants.TEXT_FIELD: record['text'],
                    Constants.USER_ITEM_KEY_FIELD: '%s|%s' % (user_id, item_id),
                }
            )

        self.records = new_records

    def add_integer_ids(self):
        users_map = {}
        items_map = {}
        user_index = 0
        item_index = 0

        for record in self.records:
            user_id = record[Constants.USER_ID_FIELD]
            item_id = record[Constants.ITEM_ID_FIELD]

            if user_id not in users_map:
                users_map[user_id] = user_index
                user_index += 1

            if item_id not in items_map:
                items_map[item_id] = item_index
                item_index += 1

            record[Constants.USER_INTEGER_ID_FIELD] = users_map[user_id]
            record[Constants.ITEM_INTEGER_ID_FIELD] = items_map[item_id]
            record[Constants.USER_ITEM_INTEGER_KEY_FIELD] = \
                '%d|%d' % (users_map[user_id], items_map[item_id])

    def tag_reviews_language(self):

        print('%s: tag reviews language' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        if os.path.exists(Constants.LANGUAGE_RECORDS_FILE):
            print('Records have already been tagged with language field')
            self.records = \
                ETLUtils.load_json_file(Constants.LANGUAGE_RECORDS_FILE)
            return

        DetectorFactory.seed = 0

        for record in self.records:
            try:
                language = langdetect.detect(record[Constants.TEXT_FIELD])
            except LangDetectException:
                language = 'unknown'
            record[Constants.LANGUAGE_FIELD] = language

        ETLUtils.save_json_file(Constants.LANGUAGE_RECORDS_FILE, self.records)

    def remove_foreign_reviews(self):

        print('%s: remove foreign reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        initial_length = len(self.records)

        self.records = ETLUtils.filter_records(
            self.records, Constants.LANGUAGE_FIELD, [Constants.LANGUAGE])
        final_length = len(self.records)
        removed_records_count = initial_length - final_length
        percentage = removed_records_count / float(initial_length) * 100

        msg = "A total of %d (%f%%) records were removed because their " \
              "language was not '%s'" % (
                removed_records_count, percentage, Constants.LANGUAGE)
        print(msg)

    def remove_users_with_low_reviews(self):
        print('%s: remove users with low reviews' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        # Remove from the dataset users with a low number of reviews
        min_reviews_per_user = Constants.MIN_REVIEWS_PER_USER
        if min_reviews_per_user is None or min_reviews_per_user < 2:
            return
        self.records = extractor.remove_users_with_low_reviews(
            self.records, min_reviews_per_user)

    def remove_items_with_low_reviews(self):
        print('%s: remove items with low reviews' % time.strftime(
            "%Y/%m/%d-%H:%M:%S"))

        # Remove from the dataset items with a low number of reviews
        min_reviews_per_item = Constants.MIN_REVIEWS_PER_ITEM
        if min_reviews_per_item is None or min_reviews_per_item < 2:
            return
        self.records = extractor.remove_items_with_low_reviews(
            self.records, min_reviews_per_item)

    def clean_reviews(self):
        print('%s: clean reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        initial_length = len(self.records)

        if Constants.ITEM_TYPE == 'fourcity_hotel':
            self.records = ETLUtils.filter_out_records(
                self.records, Constants.USER_ID_FIELD, ['', 'CATID_'])
            final_length = len(self.records)
            removed_records_count = initial_length - final_length
            percentage = removed_records_count / float(initial_length) * 100

            msg = "A total of %d (%f%%) records were removed because they " \
                  "were dirty" % (removed_records_count, percentage)
            print(msg)

    def remove_duplicate_reviews(self):
        print('%s: remove duplicate records' % time.strftime(
            "%Y/%m/%d-%H:%M:%S"))

        ids_set = set()
        non_duplicated_records = []
        initial_length = len(self.records)

        for record in self.records:
            if record[Constants.USER_ITEM_INTEGER_KEY_FIELD] not in ids_set:
                ids_set.add(record[Constants.USER_ITEM_INTEGER_KEY_FIELD])
                non_duplicated_records.append(record)

        self.records = non_duplicated_records

        final_length = len(self.records)
        removed_records_count = initial_length - final_length
        percentage = removed_records_count / float(initial_length) * 100

        msg = "A total of %d (%f%%) records were removed because they " \
              "were duplicated" % (removed_records_count, percentage)
        print(msg)

    def count_frequencies(self):
        """
        Counts the number of reviews each user and item have and stores the
        results in two separate files, one for the users and another one for the
        items. Note that the integer IDs are used and not the original user and
        item IDs
        """
        print('%s: count frequencies' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        user_frequency_map = ETLUtils.count_frequency(
            self.records, Constants.USER_INTEGER_ID_FIELD)
        item_frequency_map = ETLUtils.count_frequency(
            self.records, Constants.ITEM_INTEGER_ID_FIELD)

        user_frequency_file = Constants.generate_file_name(
            'user_frequency_map', 'json', Constants.CACHE_FOLDER, None, None,
            False
        )
        item_frequency_file = Constants.generate_file_name(
            'item_frequency_map', 'json', Constants.CACHE_FOLDER, None, None,
            False
        )

        ETLUtils.save_json_file(user_frequency_file, [user_frequency_map])
        ETLUtils.save_json_file(item_frequency_file, [item_frequency_map])

    @staticmethod
    def pos_tag_reviews(records):
        print('%s: tag reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        tagger = PerceptronTagger()

        for record in records:
            tagged_words =\
                nlp_utils.tag_words(record[Constants.TEXT_FIELD], tagger)
            record[Constants.POS_TAGS_FIELD] = tagged_words

    @staticmethod
    def lemmatize_reviews(records):
        """
        Performs a POS tagging on the text contained in the reviews and
        additionally finds the lemma of each word in the review

        :type records: list[dict]
        :param records: a list of dictionaries with the reviews
        """
        print('%s: lemmatize reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        record_index = 0
        for record in records:
            #

            tagged_words =\
                nlp_utils.lemmatize_text(record[Constants.TEXT_FIELD])

            record[Constants.POS_TAGS_FIELD] = tagged_words
            record_index += 1

        return records
        # print('')

    @staticmethod
    def lemmatize_sentences(records):
        print('%s: lemmatize sentences' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        sentence_records = []
        record_index = 0
        document_level = Constants.DOCUMENT_LEVEL
        for record in records:
            sentences = \
                nlp_utils.get_sentences(record[Constants.TEXT_FIELD])
            sentence_index = 0
            for sentence in sentences:
                if isinstance(document_level, (int, float)) and\
                        sentence_index >= document_level:
                    break
                tagged_words = nlp_utils.lemmatize_sentence(sentence)
                sentence_record = {}
                sentence_record.update(record)
                sentence_record[Constants.TEXT_FIELD] = sentence
                sentence_record['sentence_index'] = sentence_index
                sentence_record[Constants.POS_TAGS_FIELD] = tagged_words
                sentence_records.append(sentence_record)
                sentence_index += 1
                # print(sentence_record)
            record_index += 1
            # print('\rrecord index: %d/%d' % (record_index, len(records))),
        return sentence_records

    def lemmatize_records(self):

        if os.path.exists(Constants.LEMMATIZED_RECORDS_FILE):
            print('Records were already lemmatized')
            self.records = \
                ETLUtils.load_json_file(Constants.LEMMATIZED_RECORDS_FILE)
            return

        if Constants.DOCUMENT_LEVEL == 'review':
            self.records = self.lemmatize_reviews(self.records)
        elif Constants.DOCUMENT_LEVEL == 'sentence' or\
                isinstance(Constants.DOCUMENT_LEVEL, (int, long)):
            self.records = self.lemmatize_sentences(self.records)

        ETLUtils.save_json_file(Constants.LEMMATIZED_RECORDS_FILE, self.records)

    def classify_reviews(self):
        print('%s: classify reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        print(Constants.CLASSIFIED_RECORDS_FILE)
        training_records =\
            ETLUtils.load_json_file(Constants.CLASSIFIED_RECORDS_FILE)

        # If document level set to sentence (can be either 'sentence' or int)
        document_level = Constants.DOCUMENT_LEVEL
        if document_level != 'review':

            if document_level == 'sentence':
                document_level = float("inf")

            training_records = [
                record for record in training_records
                if record['sentence_index'] < document_level
            ]
            for record in training_records:
                record['specific'] = \
                    'yes' if record['sentence_type'] == 'specific' else 'no'
            print('num training records', len(training_records))

        training_records = self.lemmatize_reviews(training_records)

        classifier = ReviewsClassifier(self.classifier, self.resampler)
        classifier.train(training_records)
        classifier.label_json_reviews(self.records)

    def build_bag_of_words(self):
        print('%s: build bag of words' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        bow_type = Constants.BOW_TYPE
        cached_stop_words = set(stopwords.words("english"))
        cached_stop_words |= {
            't', 'didn', 'doesn', 'haven', 'don', 'aren', 'isn', 've', 'll',
            'couldn', 'm', 'hasn', 'hadn', 'won', 'shouldn', 's', 'wasn',
            'wouldn'}

        if Constants.LEMMATIZE:
            tagged_word_index = 2
        else:
            tagged_word_index = 0

        for record in self.records:
            bag_of_words = []
            tagged_words = record[Constants.POS_TAGS_FIELD]

            for tagged_word in tagged_words:
                if bow_type is None or tagged_word[1].startswith(bow_type):
                    bag_of_words.append(tagged_word[tagged_word_index])

            bag_of_words = [
                word for word in bag_of_words if word not in cached_stop_words
            ]

            record[Constants.BOW_FIELD] = bag_of_words

    def build_dictionary(self):
        print('%s: build dictionary' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        if self.use_cache and os.path.exists(Constants.DICTIONARY_FILE):
            print('Dictionary already exists')
            self.dictionary = corpora.Dictionary.load(Constants.DICTIONARY_FILE)

        all_words = []

        for record in self.records:
            all_words.append(record[Constants.BOW_FIELD])

        self.dictionary = corpora.Dictionary(all_words)
        sorted_words = sorted(self.dictionary.dfs.items(),
                              key=operator.itemgetter(1), reverse=True)
        for word_id, frequency in sorted_words[:100]:
            print(self.dictionary[word_id], frequency)

        self.dictionary.filter_extremes(
            Constants.MIN_DICTIONARY_WORD_COUNT,
            Constants.MAX_DICTIONARY_WORD_COUNT)

        self.dictionary.save(Constants.DICTIONARY_FILE)

    def tag_contextual_reviews(self):
        """
        Puts a tag of contextual or non-contextual to the the records that have
        contextual words (the ones that appear in the manually defined set in
        topics_analyzer.all_context_words)with those records
        """

        context_words = set(all_context_words[Constants.ITEM_TYPE])
        num_context_records = 0
        num_no_context_records = 0

        for record in self.records:
            words = set(record[Constants.BOW_FIELD])

            # If there is an intersection
            if context_words & words:
                record[Constants.HAS_CONTEXT_FIELD] = True
                num_context_records += 1
            else:
                record[Constants.HAS_CONTEXT_FIELD] = False
                num_no_context_records += 1

        print('important records: %d' % len(self.records))
        print('context records: %d' % num_context_records)
        print('no context records: %d' % num_no_context_records)

    def build_corpus(self):
        print('%s: build corpus' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        for record in self.records:
            record[Constants.CORPUS_FIELD] =\
                self.dictionary.doc2bow(record[Constants.BOW_FIELD])

    def export_records(self):
        print('%s: export records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        ETLUtils.save_json_file(
            Constants.FULL_PROCESSED_RECORDS_FILE, self.records)
        self.drop_unnecessary_fields()
        ETLUtils.save_json_file(Constants.PROCESSED_RECORDS_FILE, self.records)

    def label_review_targets(self):

        if Constants.TOPIC_MODEL_TARGET_TYPE == 'context':
            for record in self.records:
                record[Constants.TOPIC_MODEL_TARGET_FIELD] = \
                    record[Constants.PREDICTED_CLASS_FIELD]
        elif Constants.TOPIC_MODEL_TARGET_TYPE == 'sentiment':
            for record in self.records:
                rating = record[Constants.RATING_FIELD]
                sentiment = 'positive' if rating >= 4.0 else\
                    ('neutral' if rating >= 3.0 else 'negative')
                record[Constants.TOPIC_MODEL_TARGET_FIELD] = sentiment

    @staticmethod
    def find_topic_distribution(records):
        print('%s: finding topic distributions'
              % time.strftime("%Y/%m/%d-%H:%M:%S"))

        if Constants.TOPIC_MODEL_TYPE == 'lda':
            topic_model = topic_model_creator.load_topic_model(None, None)
            corpus = [record[Constants.CORPUS_FIELD] for record in records]
            lda_context_utils.update_reviews_with_topics(
                topic_model, corpus, records)
        elif Constants.TOPIC_MODEL_TYPE == 'ensemble':
            topic_extractor = NmfTopicExtractor()
            topic_extractor.load_trained_data()
            topic_extractor.update_reviews_with_topics(records)

        if Constants.TOPIC_MODEL_NORMALIZE:
            ReviewsPreprocessor.normalize_topics(records)

    @staticmethod
    def normalize_topics(records):
        print('%s: normalizing topics' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        num_topics = len(records[0][Constants.TOPICS_FIELD])

        for record in records:
            topics = record[Constants.TOPICS_FIELD]
            normalized_topics = []

            total_topics_weight = 0.0
            for topic in topics:
                topic_weight = topic[1]
                total_topics_weight += topic_weight

            if total_topics_weight > 0:
                for topic in topics:
                    normalized_topics.append(
                        (topic[0], topic[1] / total_topics_weight))
            else:
                for topic in topics:
                    normalized_topics.append(
                        (topic[0], 1.0 / num_topics))

            record[Constants.TOPICS_FIELD] = normalized_topics

    @staticmethod
    def update_context_topics(records):
        print('%s: updating records with contextual information'
              % time.strftime("%Y/%m/%d-%H:%M:%S"))

        context_extractor = ContextExtractor(records)
        context_extractor.separate_reviews()
        context_extractor.get_context_rich_topics()
        context_extractor.find_contextual_topics(records)

    def separate_recsys_topic_model_records(self):

        print('%s: separate_recsys_topic_model_records' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        num_records = len(self.records)
        topic_model_records = self.records[:num_records / 2]

        if not Constants.USE_CONTEXT:
            recsys_records = self.records[num_records / 2:]

            file_name = \
                Constants.generate_file_name(
                    'recsys_contextual_records', 'json', Constants.CACHE_FOLDER,
                    None, None, False, True)

            print('Records without context file: %s' % file_name)

            for record in recsys_records:
                record[Constants.CONTEXT_TOPICS_FIELD] = {'na': 1.0}

            ETLUtils.save_json_file(file_name, recsys_records)
            return

        topic_model_creator.train_topic_model(topic_model_records)

        if os.path.exists(Constants.RECSYS_TOPICS_PROCESSED_RECORDS_FILE):
            print('Recsys topic records have already been generated')
            recsys_records = ETLUtils.load_json_file(
                Constants.RECSYS_TOPICS_PROCESSED_RECORDS_FILE)
        else:
            recsys_records = self.records[num_records / 2:]
            self.find_topic_distribution(recsys_records)
            ETLUtils.save_json_file(
                Constants.RECSYS_TOPICS_PROCESSED_RECORDS_FILE, recsys_records)

        if os.path.exists(Constants.RECSYS_CONTEXTUAL_PROCESSED_RECORDS_FILE):
            print('Recsys contextual records have already been generated')
            print(Constants.RECSYS_CONTEXTUAL_PROCESSED_RECORDS_FILE)
            recsys_records = ETLUtils.load_json_file(
                Constants.RECSYS_CONTEXTUAL_PROCESSED_RECORDS_FILE)
        else:
            self.update_context_topics(recsys_records)
            ETLUtils.save_json_file(
                Constants.RECSYS_CONTEXTUAL_PROCESSED_RECORDS_FILE,
                recsys_records
            )

        context_transformer = ContextTransformer(recsys_records)
        context_transformer.load_data()
        context_transformer.transform_records()
        context_transformer.export_records()

    def drop_unnecessary_fields(self):
        print(
            '%s: drop unnecessary fields' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        unnecessary_fields = [
            Constants.TEXT_FIELD,
            Constants.POS_TAGS_FIELD,
            # Constants.BOW_FIELD
        ]

        ETLUtils.drop_fields(unnecessary_fields, self.records)

    def load_full_records(self):
        records_file = Constants.FULL_PROCESSED_RECORDS_FILE
        self.records = ETLUtils.load_json_file(records_file)

    def count_specific_generic_ratio(self):
        """
        Prints the proportion of specific and generic documents
        """

        specific_count = 0.0
        generic_count = 0.0

        for record in self.records:
            if record[Constants.PREDICTED_CLASS_FIELD] == 'specific':
                specific_count += 1
            if record[Constants.PREDICTED_CLASS_FIELD] == 'generic':
                generic_count += 1

        print('Specific reviews: %f%%' % (
            specific_count / len(self.records) * 100))
        print('Generic reviews: %f%%' % (
            generic_count / len(self.records) * 100))
        print('Specific reviews: %d' % specific_count)
        print('Generic reviews: %d' % generic_count)

    def export_to_triplet(self):
        print('%s: export to triplet' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        users_map = {}
        items_map = {}
        user_index = 0
        item_index = 0
        triplet_list = []

        for record in self.records:
            user_id = record[Constants.USER_ID_FIELD]
            item_id = record[Constants.ITEM_ID_FIELD]

            if user_id not in users_map:
                users_map[user_id] = user_index
                user_index += 1

            if item_id not in items_map:
                items_map[item_id] = item_index
                item_index += 1

            triplet = [users_map[user_id], items_map[item_id],
                       record[Constants.RATING_FIELD]]
            triplet_list.append(triplet)

        matrix = numpy.array(triplet_list)
        print('matrix shape', matrix.shape)

        with open(Constants.RATINGS_FILE, 'w') as f:
            numpy.savetxt(f, matrix, fmt='%d')

        print('Records exported as a triplet in: %s' % Constants.RATINGS_FILE)

    def preprocess(self):

        self.load_records()

        if 'yelp' in Constants.ITEM_TYPE:
            self.transform_yelp_records()
        elif 'fourcity' in Constants.ITEM_TYPE:
            self.transform_fourcity_records()

        self.add_integer_ids()
        self.clean_reviews()
        self.remove_duplicate_reviews()
        self.tag_reviews_language()
        self.remove_foreign_reviews()
        self.lemmatize_records()
        self.remove_users_with_low_reviews()
        self.remove_items_with_low_reviews()
        self.count_frequencies()
        self.shuffle_records()
        print('total_records: %d' % len(self.records))
        self.classify_reviews()
        self.build_bag_of_words()
        self.tag_contextual_reviews()
        # self.load_full_records()
        self.build_dictionary()
        self.build_corpus()
        self.label_review_targets()
        self.export_records()

        self.count_specific_generic_ratio()
        self.export_to_triplet()

        rda = ReviewsDatasetAnalyzer(self.records)
        print('density: %f' % rda.calculate_density_approx())
        print('sparsity: %f' % rda.calculate_sparsity_approx())
        print('total_records: %d' % len(self.records))
        user_ids = \
            extractor.get_groupby_list(self.records, Constants.USER_ID_FIELD)
        item_ids = \
            extractor.get_groupby_list(self.records, Constants.ITEM_ID_FIELD)
        print('total users', len(user_ids))
        print('total items', len(item_ids))

    def full_cycle(self):
        Constants.print_properties()
        print('%s: full cycle' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        utilities.plant_seeds()

        if self.use_cache and \
                os.path.exists(Constants.PROCESSED_RECORDS_FILE):
            print('Records have already been processed')
            self.records = \
                ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)
        else:
            self.preprocess()

        if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
            self.separate_recsys_topic_model_records()


def main():
    reviews_preprocessor = ReviewsPreprocessor(use_cache=True)
    reviews_preprocessor.full_cycle()

# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
