import os
import string
import time

import unicodedata

import shutil

import numpy

from etl import ETLUtils
from topicmodeling.nmf_topic_extractor import NmfTopicExtractor
from utils import utilities
from utils.constants import Constants
import utils.utilities

CARSKIT_WORKSPACE_FOLDER = Constants.CARSKIT_RATINGS_FOLDER + 'CARSKit.Workspace/'
CSV_FILE = Constants.CARSKIT_RATINGS_FOLDER + 'ratings.csv'


def get_topic_terms(topic_string):
    terms = topic_string.split(" + ")
    return [term.partition("*")[2] for term in terms]


def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters).lower()


def copy_to_workspace(file_path):
    ratings_binary_file = CARSKIT_WORKSPACE_FOLDER + 'ratings_binary.txt'
    shutil.copy(file_path, ratings_binary_file)


def find_predefined_context(word_list):

    context_map = utils.utilities.context_words[Constants.ITEM_TYPE]
    existing_context_list = []

    for context_category, context_words in context_map.items():
        if context_words & set(word_list):
            existing_context_list.append(context_category)

    return existing_context_list


def extract_topic_id(topic_name):
    topic_id = topic_name.split('topic_')

    if len(topic_id) == 2:
        return int(topic_id[1])
    else:
        return None


def create_topic_categories_map(topic_ids, topic_extractor):
    topic_categories_map = {}

    for topic_id in topic_ids:
        topic_terms = get_topic_terms(topic_extractor.print_topic(topic_id))
        topic_categories = find_predefined_context(topic_terms)
        topic_categories_map[topic_id] = topic_categories

    return topic_categories_map


class CarsKitExporter:

    def __init__(self, topics_field=Constants.CONTEXT_TOPICS_FIELD):
        self.topic_extractor = None
        self.records = None
        self.topics_field = topics_field

    def load_data(self):
        """
        Loads the records and the topic model from files

        """
        self.records = ETLUtils.load_json_file(
            Constants.RECSYS_TOPICS_PROCESSED_RECORDS_FILE)
        self.topic_extractor = NmfTopicExtractor()
        self.topic_extractor.load_trained_data()

    def print_topics(self):
        return self.topic_extractor.print_topic_model()

    def export(self):

        if not os.path.exists(Constants.CARSKIT_RATINGS_FOLDER):
            os.makedirs(Constants.CARSKIT_RATINGS_FOLDER)
        if not os.path.exists(CARSKIT_WORKSPACE_FOLDER):
            os.makedirs(CARSKIT_WORKSPACE_FOLDER)

        json_ratings_file = Constants.CARSKIT_RATINGS_FOLDER + 'ratings.json'
        shutil.copy(
            Constants.RECSYS_CONTEXTUAL_PROCESSED_RECORDS_FILE,
            json_ratings_file
        )

        carskit_format = Constants.CARSKIT_NOMINAL_FORMAT

        if carskit_format == 'no_context':
            self.export_without_context()
        elif carskit_format == 'top_words':
            self.export_as_top_word()
        elif carskit_format == 'predefined_context':
            self.export_as_predefined_context()
        elif carskit_format == 'topic_predefined_context':
            self.export_as_topic_predefined_context()
        else:
            raise ValueError('%s option does not exist' % carskit_format)

    def export_without_context(self):
        print('%s: exporting to CARSKit binary ratings format without context' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        if os.path.exists(CSV_FILE):
            print('Binary ratings file already exists')
            copy_to_workspace(CSV_FILE)
            return

        new_records = []
        numpy.random.seed(0)

        for record in self.records:

            context_na_value = 1

            new_records.append({
                Constants.USER_ID_FIELD: record[Constants.USER_INTEGER_ID_FIELD],
                Constants.ITEM_ID_FIELD: record[Constants.ITEM_INTEGER_ID_FIELD],
                Constants.RATING_FIELD: record[Constants.RATING_FIELD],
                'context:na': context_na_value,
            })

        headers = [
            Constants.USER_ID_FIELD,
            Constants.ITEM_ID_FIELD,
            Constants.RATING_FIELD,
            'context:na'
        ]

        ETLUtils.save_csv_file(CSV_FILE, new_records, headers)
        copy_to_workspace(CSV_FILE)

    def export_as_top_word(self):
        print('%s: exporting to CARSKit ratings binary format with context as '
              'top words' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        if os.path.exists(CSV_FILE):
            print('Binary ratings file already exists')
            copy_to_workspace(CSV_FILE)
            return

        new_records = []
        topic_model_string = self.topic_extractor.print_topic_model()
        top_terms = [get_topic_terms(topic) for topic in topic_model_string]
        context_headers = ['context:%s' % term[0] for term in top_terms]

        for record in self.records:

            new_record = {
                Constants.USER_ID_FIELD: record[Constants.USER_INTEGER_ID_FIELD],
                Constants.ITEM_ID_FIELD: record[Constants.ITEM_INTEGER_ID_FIELD],
                Constants.RATING_FIELD: record[Constants.RATING_FIELD],
            }

            topics = record[self.topics_field]
            context_found = False

            for topic in topics:
                topic_index = topic[0]
                topic_weight = topic[1]

                context_key = context_headers[topic_index]
                context_value = 1 if topic_weight > 0.0 else 0

                new_record[context_key] = context_value
            # print(new_record)
            context_na_value = 0 if context_found else 1
            new_record['context:na'] = context_na_value

            new_records.append(new_record)

        headers = [
            Constants.USER_ID_FIELD,
            Constants.ITEM_ID_FIELD,
            Constants.RATING_FIELD,
            'context:na'
        ]
        headers.extend(context_headers)
        ETLUtils.save_csv_file(CSV_FILE, new_records, headers)
        copy_to_workspace(CSV_FILE)

    def export_as_predefined_context(self):
        print('%s: exporting to CARSKit ratings binary format with context as '
              'predefined context' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        if os.path.exists(CSV_FILE):
            print('Binary ratings file already exists')
            copy_to_workspace(CSV_FILE)
            return

        new_records = []

        context_categories = utilities.context_words[Constants.ITEM_TYPE].keys()
        context_headers = [
            'context:%s' % category for category in context_categories]

        index = 0

        for record in self.records:

            new_record = {
                Constants.USER_ID_FIELD: record[Constants.USER_INTEGER_ID_FIELD],
                Constants.ITEM_ID_FIELD: record[Constants.ITEM_INTEGER_ID_FIELD],
                Constants.RATING_FIELD: record[Constants.RATING_FIELD],
            }

            review_categories = \
                find_predefined_context(record[Constants.BOW_FIELD])

            context_found = False
            for category in context_categories:
                category_key = 'context:' + category
                category_value = 0
                if category in review_categories:
                    category_value = 1
                    context_found = True
                new_record[category_key] = category_value

            context_na_value = 0 if context_found else 1
            new_record['context:na'] = context_na_value

            new_records.append(new_record)
            index += 1

        headers = [
            Constants.USER_ID_FIELD,
            Constants.ITEM_ID_FIELD,
            Constants.RATING_FIELD,
            'context:na'
        ]
        headers.extend(context_headers)
        ETLUtils.save_csv_file(CSV_FILE, new_records, headers)
        copy_to_workspace(CSV_FILE)

    def export_as_topic_predefined_context(self):
        print('%s: exporting to CARSKit ratings binary format with context as '
              'predefined context' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        if os.path.exists(CSV_FILE):
            print('Binary ratings file already exists')
            copy_to_workspace(CSV_FILE)
            return

        new_records = []

        context_categories = utilities.context_words[Constants.ITEM_TYPE].keys()
        context_headers = [
            'context:%s' % category for category in context_categories]
        context_topic_ids = [
            extract_topic_id(topic_name) for topic_name in
            self.records[0][Constants.CONTEXT_TOPICS_FIELD].keys()]
        context_topic_ids = [
            topic for topic in context_topic_ids if topic is not None]
        topic_categories_map = \
            create_topic_categories_map(context_topic_ids, self.topic_extractor)

        print(topic_categories_map)

        index = 0

        # for record in self.records[3:4]:
        for record in self.records:

            new_record = {
                Constants.USER_ID_FIELD: record[Constants.USER_INTEGER_ID_FIELD],
                Constants.ITEM_ID_FIELD: record[Constants.ITEM_INTEGER_ID_FIELD],
                Constants.RATING_FIELD: record[Constants.RATING_FIELD],
            }

            context_topics = record[Constants.CONTEXT_TOPICS_FIELD]

            # topic_categories = \
            #     find_predefined_context(record[Constants.BOW_FIELD])

            context_found = False
            for category in context_categories:
                category_key = 'context:' + category
                category_value = 0

                for topic_name in context_topics.keys():
                    topic_id = extract_topic_id(topic_name)
                    if topic_id is None:
                        break
                    topic_categories = topic_categories_map[topic_id]
                    if context_topics[topic_name] > 0 and category in topic_categories:
                        category_value = 1
                        context_found = True
                new_record[category_key] = category_value

            context_na_value = 0 if context_found else 1
            new_record['context:na'] = context_na_value

            new_records.append(new_record)
            index += 1

        headers = [
            Constants.USER_ID_FIELD,
            Constants.ITEM_ID_FIELD,
            Constants.RATING_FIELD,
            'context:na'
        ]
        headers.extend(context_headers)

        print(new_records[0])
        # print(new_records[10])
        # print(new_records[100])

        # record_index = 0
        # all_context_headers = context_headers + ['context:na']
        # for record in new_records:
        #
        #     context_sum = 0
        #     for header in all_context_headers:
        #         context_sum += record[header]
        #     record_index += 1
        #     print(record_index, context_sum)

        ETLUtils.save_csv_file(CSV_FILE, new_records, headers)
        copy_to_workspace(CSV_FILE)

    def export_as_all_words(self):
        print('%s: exporting to CARSKit ratings binary format with context as '
              'all words' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        if os.path.exists(CSV_FILE):
            print('Binary ratings file already exists')
            copy_to_workspace(CSV_FILE)
            return

        new_records = []
        all_terms = set()
        for record in self.records:
            all_terms |= set(record[Constants.BOW_FIELD])

        all_terms = [remove_accents(term) for term in all_terms]

        context_headers = ['context:%s' % term for term in all_terms]

        for record in self.records:

            new_record = {
                Constants.USER_ID_FIELD: record[Constants.USER_INTEGER_ID_FIELD],
                Constants.ITEM_ID_FIELD: record[Constants.ITEM_INTEGER_ID_FIELD],
                Constants.RATING_FIELD: record[Constants.RATING_FIELD],
            }

            bag_of_words = record[Constants.BOW_FIELD]

            for term, context_header in zip(all_terms, context_headers):
                context_value = 1 if term in bag_of_words > 0.0 else 0

                new_record[context_header] = context_value

            new_records.append(new_record)

        headers = [
            Constants.USER_ID_FIELD,
            Constants.ITEM_ID_FIELD,
            Constants.RATING_FIELD,
        ]
        headers.extend(context_headers)
        print(len(headers))
        print(headers)
        ETLUtils.save_csv_file(CSV_FILE, new_records, headers)
        copy_to_workspace(CSV_FILE)


def main():

    carskit_exporter = CarsKitExporter(Constants.TOPICS_FIELD)
    carskit_exporter.load_data()
    # string_topics = carskit_exporter.print_topics()
    # for topic in string_topics:
    #     print(topic)
    # carskit_exporter.export_without_context()
    # carskit_exporter.export_as_top_word()
    # carskit_exporter.export_as_all_words()
    # carskit_exporter.export_as_topic_predefined_context()
    carskit_exporter.export()

    #
    # my_words = [
    #     'wedding',
    #     'casino',
    # ]
    # print(find_predefined_context(my_words))


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)





