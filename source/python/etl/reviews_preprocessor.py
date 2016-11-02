import json
import os
import random

import time

import operator

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
from unbalanced_dataset.combine import SMOTEENN
from unbalanced_dataset.combine import SMOTETomek
from unbalanced_dataset.over_sampling import RandomOverSampler
from unbalanced_dataset.over_sampling import SMOTE

from etl import ETLUtils
from nlp import nlp_utils
from topicmodeling.context import topic_model_creator
from topicmodeling.context.reviews_classifier import ReviewsClassifier
from tripadvisor.fourcity import extractor
from utils import utilities
from utils.constants import Constants


class ReviewsPreprocessor:

    def __init__(self, use_cache=False):
        self.use_cache = use_cache
        self.records = None
        self.dictionary = None

        ratio = 'auto'
        verbose = False
        resampler = Constants.RESAMPLER
        classifier = Constants.DOCUMENT_CLASSIFIER
        random_state = Constants.DOCUMENT_CLASSIFIER_SEED
        classifiers = {
            'logistic_regression': LogisticRegression(C=100),
            'svc': SVC(),
            'kneighbors': KNeighborsClassifier(n_neighbors=10),
            'decision_tree': DecisionTreeClassifier(),
            'nu_svc': NuSVC(),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        samplers = {
            'random_over_sampler': RandomOverSampler(
                ratio, random_state=random_state, verbose=verbose),
            'smote_regular': SMOTE(
                ratio, random_state=random_state, verbose=verbose,
                kind='regular'),
            'smote_bl1': SMOTE(
                ratio, random_state=random_state, verbose=verbose,
                kind='borderline1'),
            'smote_bl2': SMOTE(
                ratio, random_state=random_state, verbose=verbose,
                kind='borderline2'),
            'smote_tomek': SMOTETomek(
                ratio, random_state=random_state, verbose=verbose),
            'smote-enn': SMOTEENN(
                ratio, random_state=random_state, verbose=verbose)
        }
        self.classifier = classifiers[classifier]
        self.resampler = samplers[resampler]
        classifiers = None
        samplers = None

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
            new_records.append(
                {
                    Constants.REVIEW_ID_FIELD: record['review_id'],
                    Constants.USER_ID_FIELD: record['user_id'],
                    Constants.ITEM_ID_FIELD: record['business_id'],
                    Constants.RATING_FIELD: record['stars'],
                    Constants.TEXT_FIELD: record['text'],

                }
            )

        self.records = new_records

    def transform_fourcity_records(self):
        new_records = []

        for record in self.records:
            new_records.append(
                {
                    Constants.REVIEW_ID_FIELD: record['id'],
                    Constants.USER_ID_FIELD: record['author']['id'],
                    Constants.ITEM_ID_FIELD: record['offering_id'],
                    Constants.RATING_FIELD: record['ratings']['overall'],
                    Constants.TEXT_FIELD: record['text'],

                }
            )

        self.records = new_records

    def remove_users_with_low_reviews(self):

        # Remove from the dataset users with a low number of reviews
        min_reviews_per_user = Constants.MIN_REVIEWS_PER_USER
        if min_reviews_per_user is None or min_reviews_per_user < 2:
            return
        self.records = extractor.remove_users_with_low_reviews(
            self.records, min_reviews_per_user)

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

        if Constants.DOCUMENT_LEVEL == 'review':
            self.records = self.lemmatize_reviews(self.records)
        elif Constants.DOCUMENT_LEVEL == 'sentence' or\
                isinstance(Constants.DOCUMENT_LEVEL, (int, long)):
            self.records = self.lemmatize_sentences(self.records)

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
            return corpora.Dictionary.load(Constants.DICTIONARY_FILE)

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

    def create_user_item_map(self):
        print('%s: creating user-item map, this could take a long time'
              % time.strftime("%Y/%m/%d-%H:%M:%S"))

        print('user-item map file: %s' % Constants.USER_ITEM_MAP_FILE)

        if os.path.exists(Constants.USER_ITEM_MAP_FILE):
            print('User-item map already exists')
            with open(Constants.USER_ITEM_MAP_FILE, 'r') as fp:
                user_item_map = json.load(fp)
            return user_item_map

        user_ids = extractor.get_groupby_list(self.records, Constants.USER_ID_FIELD)
        user_item_map = {}

        for user_id in user_ids:
            user_records = ETLUtils.filter_records(
                self.records, Constants.USER_ID_FIELD, [user_id])
            user_items = extractor.get_groupby_list(
                user_records, Constants.ITEM_ID_FIELD)
            user_item_map[user_id] = user_items

        with open(Constants.USER_ITEM_MAP_FILE, 'w') as fp:
            json.dump(user_item_map, fp)

        return user_item_map

    def build_corpus(self):
        print('%s: build corpus' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        for record in self.records:
            record[Constants.CORPUS_FIELD] =\
                self.dictionary.doc2bow(record[Constants.BOW_FIELD])
        ETLUtils.save_json_file(
            Constants.FULL_PROCESSED_RECORDS_FILE, self.records)
        self.drop_unnecessary_fields()
        ETLUtils.save_json_file(Constants.PROCESSED_RECORDS_FILE, self.records)

    def separate_recsys_topic_model_records(self):

        print('%s: separate_recsys_topic_model_records' %
            time.strftime("%Y/%m/%d-%H:%M:%S"))

        num_records = len(self.records)
        topic_model_records = self.records[:num_records/2]
        recsys_records = self.records[num_records/2:]
        topic_model = topic_model_creator.create_topic_model(
            topic_model_records, None, None)

        recsys_records_file = utilities.generate_file_name(
            'recsys_records', 'json', Constants.CACHE_FOLDER, None, None, False)

        if self.use_cache and os.path.exists(recsys_records_file):
            print('Recsys records have already been generated')
            return

        recsys_topics_records_file = utilities.generate_file_name(
            'recsys_topics_records', 'json', Constants.CACHE_FOLDER, None, None,
            True)

        topic_model.find_contextual_topics(recsys_records)
        ETLUtils.save_json_file(recsys_topics_records_file, recsys_records)

        topics_file_path = utilities.generate_file_name(
            'context_topics', 'json', Constants.CACHE_FOLDER, None, None, True)
        ETLUtils.save_json_file(
            topics_file_path, [dict(topic_model.context_rich_topics)])

    def drop_unnecessary_fields(self):
        print(
            '%s: drop unnecessary fields' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        unnecessary_fields = [
            Constants.TEXT_FIELD,
            Constants.POS_TAGS_FIELD,
            # Constants.VOTES_FIELD,
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

    def full_cycle(self):
        Constants.print_properties()
        print('%s: full cycle' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        utilities.plant_seeds()

        if self.use_cache and os.path.exists(Constants.FULL_PROCESSED_RECORDS_FILE):
            print('Records have already been processed')
            self.records = ETLUtils.load_json_file(Constants.FULL_PROCESSED_RECORDS_FILE)
            self.drop_unnecessary_fields()
            ETLUtils.save_json_file(
                Constants.PROCESSED_RECORDS_FILE, self.records)
        else:
            self.load_records()

            if 'yelp' in Constants.ITEM_TYPE:
                self.transform_yelp_records()
            elif 'fourcity' in Constants.ITEM_TYPE:
                self.transform_fourcity_records()

            self.shuffle_records()
            self.remove_users_with_low_reviews()
            self.lemmatize_records()
            print('total_records: %d' % len(self.records))
            self.classify_reviews()
            self.build_bag_of_words()
            # self.load_full_records()
            self.build_dictionary()
            self.build_corpus()

        self.create_user_item_map()
        self.count_specific_generic_ratio()
        self.export_to_triplet()

        if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
            self.separate_recsys_topic_model_records()


def main():
    reviews_preprocessor = ReviewsPreprocessor(use_cache=True)
    reviews_preprocessor.full_cycle()

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
