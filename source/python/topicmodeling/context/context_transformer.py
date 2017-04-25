import time

from etl import ETLUtils
from topicmodeling.nmf_topic_extractor import NmfTopicExtractor
from utils import utilities
from utils.constants import Constants


def get_topic_terms(topic_string):
    terms = topic_string.split(" + ")
    return [term.partition("*")[2] for term in terms]


def find_predefined_context(word_list):

    context_map = utilities.context_words[Constants.ITEM_TYPE]
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


class ContextTransformer:

    def __init__(self, records):
        self.topic_extractor = None
        self.records = records

    def load_data(self):
        """
        Loads the records and the topic model from files

        """
        self.topic_extractor = NmfTopicExtractor()
        self.topic_extractor.load_trained_data()

    def transform_records(self):

        context_format = Constants.CONTEXT_FORMAT

        if context_format == 'no_context':
            self.transform_without_context()
        elif context_format == 'context_topic_weights':
            self.transform_as_context_topic_weights()
        elif context_format == 'top_words':
            self.transform_as_top_words()
        elif context_format == 'predefined_context':
            self.transform_as_predefined_context()
        elif context_format == 'topic_predefined_context':
            self.transform_as_topic_predefined_context()
        else:
            raise ValueError('%s option does not exist' % context_format)

    def transform_without_context(self):
        print('%s: transforming records to records without context' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        for record in self.records:
            record[Constants.CONTEXT_FIELD] = {Constants.EMPTY_CONTEXT: 1.0}

    def transform_as_context_topic_weights(self):
        print('%s: transforming records as context topic weigths' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        for record in self.records:
            context_topics = record[Constants.CONTEXT_TOPICS_FIELD].copy()

            if sum(context_topics.values()) > 0:
                context_topics[Constants.EMPTY_CONTEXT] = 0.0
            else:
                context_topics[Constants.EMPTY_CONTEXT] = 1.0

            record[Constants.CONTEXT_FIELD] = context_topics

    def transform_as_top_words(self):
        print('%s: transforming records as top words' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        topic_model_string = self.topic_extractor.print_topic_model()
        top_words = [get_topic_terms(topic)[0] for topic in topic_model_string]

        for record in self.records:
            context_topics = record[Constants.CONTEXT_TOPICS_FIELD]
            context_map = {}
            context_na_value = 1

            for topic_name, topic_weight in context_topics.items():

                if topic_name == 'nocontexttopics':
                    continue

                topic_index = extract_topic_id(topic_name)

                context_key = top_words[topic_index]
                if topic_weight > 0.0:
                    context_value = 1
                    context_na_value = 0
                else:
                    context_value = 0
                context_map[context_key] = context_value

            context_map[Constants.EMPTY_CONTEXT] = context_na_value
            record[Constants.CONTEXT_FIELD] = context_map

    def transform_as_predefined_context(self):
        print('%s: transforming records as predefined context' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        context_categories = utilities.context_words[Constants.ITEM_TYPE].keys()

        for record in self.records:

            review_categories = \
                find_predefined_context(record[Constants.BOW_FIELD])
            context_map = {}

            context_na_value = 1
            for category in context_categories:
                category_value = 0
                if category in review_categories:
                    category_value = 1
                    context_na_value = 0
                context_map[category] = category_value

            context_map[Constants.EMPTY_CONTEXT] = context_na_value
            record[Constants.CONTEXT_FIELD] = context_map

    def transform_as_topic_predefined_context(self):
        print('%s: transforming records as topic predefined context' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        context_categories = utilities.context_words[Constants.ITEM_TYPE].keys()
        context_topic_ids = [
            extract_topic_id(topic_name) for topic_name in
            self.records[0][Constants.CONTEXT_TOPICS_FIELD].keys()]
        context_topic_ids = [
            topic for topic in context_topic_ids if topic is not None]
        topic_categories_map = \
            create_topic_categories_map(context_topic_ids, self.topic_extractor)

        for record in self.records:

            context_topics = record[Constants.CONTEXT_TOPICS_FIELD]
            context_map = {}

            context_na_value = 1
            for category in context_categories:
                category_value = 0

                for topic_name in context_topics.keys():
                    topic_id = extract_topic_id(topic_name)
                    if topic_id is None:
                        break
                    topic_categories = topic_categories_map[topic_id]
                    if context_topics[topic_name] > 0 and category in topic_categories:
                        category_value = 1
                        context_na_value = 0
                context_map[category] = category_value

            context_map[Constants.EMPTY_CONTEXT] = context_na_value
            record[Constants.CONTEXT_FIELD] = context_map

    def export_records(self):
        print('%s: exporting transformed records' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        records_to_export = []
        desired_fields = [
            Constants.USER_INTEGER_ID_FIELD,
            Constants.ITEM_INTEGER_ID_FIELD,
            Constants.RATING_FIELD,
            Constants.CONTEXT_FIELD,
        ]

        for record in self.records:
            new_record = {field: record[field] for field in desired_fields}
            records_to_export.append(new_record)

        file_name = Constants.generate_file_name(
            'recsys_formatted_context_records', 'json', Constants.CACHE_FOLDER,
            None, None, True, True, uses_carskit=False, normalize_topics=True,
            format_context=True)
        ETLUtils.save_json_file(file_name, records_to_export)


def main():

    records = ETLUtils.load_json_file(
        Constants.RECSYS_CONTEXTUAL_PROCESSED_RECORDS_FILE)
    context_transformer = ContextTransformer(records)
    context_transformer.load_data()
    context_transformer.transform_records()
    context_transformer.export_records()

    # context_transformer.transform_without_context()
    # context_transformer.transform_as_context_topic_weights()
    # context_transformer.transform_as_top_words()
    # context_transformer.transform_as_predefined_context()
    # context_transformer.transform_as_topic_predefined_context()

    # for record in records:
        # if record[Constants.CONTEXT_FIELD][Constants.EMPTY_CONTEXT] == 0:
        #     continue
        # print(record[Constants.CONTEXT_FIELD], record[Constants.CONTEXT_TOPICS_FIELD])

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
