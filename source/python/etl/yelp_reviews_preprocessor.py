import random

import time

import numpy
import operator
from gensim import corpora
from nltk import PerceptronTagger
from nltk.corpus import stopwords

from etl import ETLUtils
from nlp import nlp_utils
from topicmodeling.context.reviews_classifier import ReviewsClassifier
from utils.constants import Constants


class YelpReviewsPreprocessor:

    def __init__(self):
        self.records = None
        self.dictionary = None

    @staticmethod
    def plant_seeds():
        print('%s: plant seeds' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        random.seed(0)
        numpy.random.seed(0)

    def load_records(self):
        print('%s: load records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        records_file =\
            Constants.DATASET_FOLDER + 'yelp_training_set_review_' +\
            Constants.ITEM_TYPE + 's.json'
        self.records = ETLUtils.load_json_file(records_file)

    def shuffle_records(self):
        print('%s: shuffle records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        random.shuffle(self.records)

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
        print('%s: lemmatize reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        record_index = 0
        max_sentences = 1
        for record in records:

            print('\rrecord index: %d/%d' % (record_index, len(records))),

            if Constants.MAX_SENTENCES is None:
                tagged_words =\
                    nlp_utils.lemmatize_text(record[Constants.TEXT_FIELD])
            else:
                sentences = nlp_utils.get_sentences(record[Constants.TEXT_FIELD])
                sentence_index = 0
                tagged_words = []
                for sentence in sentences:
                    # print(sentence_index, sentence)
                    if sentence_index >= max_sentences:
                        break
                    tagged_words.extend(nlp_utils.lemmatize_sentence(sentence))
                    # print(tagged_words)
                    sentence_index += 1

            record[Constants.POS_TAGS_FIELD] = tagged_words
            record_index += 1
        print('')

    # def classify_reviews(self):
    #     print('%s: classify reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    #     dataset = Constants.ITEM_TYPE
    #     folder = Constants.DATASET_FOLDER
    #     training_records_file = folder +\
    #         'classified_' + dataset + '_reviews.json'
    #     training_records = ETLUtils.load_json_file(training_records_file)
    #     self.lemmatize_reviews(training_records)
    #
    #     classifier = ReviewsClassifier()
    #     classifier.train(training_records)
    #     classifier.label_json_reviews(self.records)

    def classify_reviews(self):
        print('%s: classify reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        dataset = Constants.ITEM_TYPE
        folder = Constants.DATASET_FOLDER
        file_name_suffix =\
            '' if Constants.MAX_SENTENCES is None else '_sentences'
        training_records_file = folder +\
            'classified_' + dataset + '_reviews' + file_name_suffix + '.json'
        training_records = ETLUtils.load_json_file(training_records_file)

        if Constants.MAX_SENTENCES is not None:
            training_records = [
                record for record in training_records
                if record['sentence_index'] < Constants.MAX_SENTENCES
            ]
            for record in training_records:
                record['specific'] = \
                    'yes' if record['sentence_type'] == 'specific' else 'no'
            print('num training records', len(training_records))

        self.lemmatize_reviews(training_records)

        classifier = ReviewsClassifier()
        classifier.train(training_records)
        classifier.label_json_reviews(self.records)

    def build_bag_of_words(self):
        print('%s: build bag of words' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        bow_type = Constants.BOW_TYPE
        cached_stop_words = set(stopwords.words("english"))

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

    def build_corpus(self):
        print('%s: build corpus' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        for record in self.records:
            record[Constants.CORPUS_FIELD] =\
                self.dictionary.doc2bow(record[Constants.BOW_FIELD])

    def export_records(self):
        print('%s: export records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        self.dictionary.save(Constants.DICTIONARY_FILE)
        ETLUtils.save_json_file(
            Constants.FULL_PROCESSED_RECORDS_FILE, self.records)
        self.drop_unnecessary_fields()
        ETLUtils.save_json_file(Constants.PROCESSED_RECORDS_FILE, self.records)

    def drop_unnecessary_fields(self):
        print('%s: drop unnecessary fields' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        unnecessary_fields = [
            Constants.TEXT_FIELD,
            Constants.POS_TAGS_FIELD,
            Constants.VOTES_FIELD,
            Constants.BOW_FIELD
        ]

        ETLUtils.drop_fields(unnecessary_fields, self.records)

    def load_full_records(self):
        records_file = Constants.FULL_PROCESSED_RECORDS_FILE
        self.records = ETLUtils.load_json_file(records_file)

    def count_specific_generic_ratio(self):
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

    def full_cycle(self):
        print(Constants._properties)
        print('%s: full cycle' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        self.plant_seeds()
        self.load_records()
        self.shuffle_records()
        # self.pos_tag_reviews(self.records)
        self.lemmatize_reviews(self.records)
        for record in self.records[:10]:
            print(record[Constants.POS_TAGS_FIELD])
        self.classify_reviews()
        self.build_bag_of_words()

        # self.load_full_records()

        self.build_dictionary()
        self.build_corpus()
        self.export_records()
        self.count_specific_generic_ratio()


def main():
    reviews_preprocessor = YelpReviewsPreprocessor()
    reviews_preprocessor.full_cycle()

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)


# 1. Shuffle the records
# 2. Tag the reviews as specific or generic
# 3. Extract the nouns, and add them as a bag of words in the JSON file
# 4. Drop any unnecessary fields
# 5. Build dictionary using only nouns and save dictionary

