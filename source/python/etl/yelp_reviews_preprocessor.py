import random

import time

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

    def load_records(self):
        print('%s: load records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        records_file =\
            Constants.DATASET_FOLDER + 'yelp_training_set_review_' +\
            Constants.ITEM_TYPE + 's.json'
        self.records = ETLUtils.load_json_file(records_file)

    def shuffle_records(self):
        print('%s: shuffle records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        random.shuffle(self.records)

    def pos_tag_reviews(self, records):
        print('%s: tag reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        tagger = PerceptronTagger()

        for record in records:
            tagged_words =\
                nlp_utils.tag_words(record[Constants.TEXT_FIELD], tagger)
            record[Constants.POS_TAGS_FIELD] = tagged_words

    def classify_reviews(self):
        print('%s: classify reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        dataset = Constants.ITEM_TYPE
        folder = Constants.DATASET_FOLDER
        training_records_file = folder +\
            'classified_' + dataset + '_reviews.json'
        training_records = ETLUtils.load_json_file(training_records_file)
        self.pos_tag_reviews(training_records)

        classifier = ReviewsClassifier()
        classifier.train(training_records)
        classifier.label_json_reviews(self.records)

    def build_bag_of_words(self):
        print('%s: build bag of words' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        bow_type = Constants.BOW_TYPE
        cached_stop_words = set(stopwords.words("english"))

        for record in self.records:
            bag_of_words = []
            tagged_words = record[Constants.POS_TAGS_FIELD]

            for tagged_word in tagged_words:
                if bow_type is None or tagged_word[1].startswith(bow_type):
                    bag_of_words.append(tagged_word[0])

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

        dfs = self.dictionary.dfs
        sorted_dict = sorted(dfs.items(), key=operator.itemgetter(1), reverse=True)

        print(self.dictionary.dfs)
        print(sorted_dict)
        print(self.dictionary.id2token)

        for word_id, word_frequency in sorted_dict[:100]:
            print(self.dictionary[word_id], word_frequency)
            # print(self.dictionary.id2token[word_id], word_frequency)

        self.dictionary.filter_extremes(2, 0.4)

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

    def full_cycle(self):
        print('%s: full cycle' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        random.seed(0)
        self.load_records()
        self.shuffle_records()
        self.pos_tag_reviews(self.records)
        self.classify_reviews()
        self.build_bag_of_words()
        self.build_dictionary()
        self.build_corpus()
        self.export_records()
        print(self.records[0])
        print(self.records[1])


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

