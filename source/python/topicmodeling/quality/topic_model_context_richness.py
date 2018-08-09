import time

import itertools
from pandas import DataFrame

from etl import ETLUtils
from topicmodeling.context import topic_model_creator
from topicmodeling.context.context_extractor import ContextExtractor
from topicmodeling.nmf_topic_extractor import NmfTopicExtractor
from utils import utilities
from utils.constants import Constants
from utils.utilities import grouped_hotel_context_words, \
    grouped_restaurant_context_words


def analyze_topics():

    start_time = time.time()

    utilities.plant_seeds()
    records = \
        ETLUtils.load_json_file(Constants.RECSYS_TOPICS_PROCESSED_RECORDS_FILE)
    print('num_reviews', len(records))
    num_topics = Constants.TOPIC_MODEL_NUM_TOPICS
    num_terms = Constants.TOPIC_MODEL_STABILITY_NUM_TERMS

    topic_model_string = None
    if Constants.TOPIC_MODEL_TYPE == 'ensemble':
        topic_model = NmfTopicExtractor()
        topic_model.load_trained_data()
        topic_model_string = topic_model.print_topic_model('max')
    elif Constants.TOPIC_MODEL_TYPE == 'lda':
        topic_model = topic_model_creator.load_topic_model(None, None)
        topic_model_string = [
            topic_model.print_topic(topic_id, num_terms)
            for topic_id in range(num_topics)
        ]
    context_extractor = ContextExtractor(records)
    context_extractor.separate_reviews()
    context_extractor.get_context_rich_topics()

    topic_data = []

    for topic in range(num_topics):
        result = {}
        result['topic_id'] = topic
        result.update(split_topic(topic_model_string[topic]))
        result['ratio'] = context_extractor.topic_ratio_map[topic]
        result['weighted_frequency'] = \
            context_extractor.topic_weighted_frequency_map[topic]
        topic_data.append(result)

    data_frame = DataFrame.from_dict(topic_data)
    scores = {}
    scores['num_topics'] = Constants.TOPIC_MODEL_NUM_TOPICS
    probability_score = data_frame['probability_score'].mean()
    scores['probability_score'] = probability_score

    print('probability score: %f' % scores['probability_score'])

    end_time = time.time()
    cycle_time = end_time - start_time
    scores['cycle_time'] = cycle_time

    print("Cycle time = %f seconds" % cycle_time)

    return scores


def split_topic(topic_string):
    """
    Splits a topic into dictionary containing each word

    :type topic_string: str
    :param topic_string:
    """

    my_context_words = []
    if 'hotel' in Constants.ITEM_TYPE:
        for values in grouped_hotel_context_words.values():
            my_context_words.extend(values)
    elif 'restaurant' in Constants.ITEM_TYPE:
        for values in grouped_restaurant_context_words.values():
            my_context_words.extend(values)

    words_dict = {}
    index = 0
    topic_words = topic_string.split(' + ')
    probability_score = 0.0
    for topic_word in topic_words:
        word = topic_word.split('*')[1]
        word_probability_score = float(topic_word.split('*')[0])
        words_dict['word' + str(index)] = topic_word.encode('utf-8')
        if word in my_context_words:
            probability_score += word_probability_score
        index += 1

    words_dict['probability_score'] = probability_score

    return words_dict


def full_cycle():

    num_topics_list = [5, 10, 20, 40]
    # bow_type_list = [None, 'NN', 'JJ', 'VB']
    review_type_list = ['specific', 'generic']
    # num_topics_list = [10]
    bow_type_list = ['NN']
    results = []

    for num_topics, bow_type, review_type in itertools.product(
            num_topics_list, bow_type_list, review_type_list):

        Constants.update_properties({
            Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: num_topics,
            Constants.BOW_TYPE_FIELD: bow_type,
            Constants.TOPIC_MODEL_TARGET_REVIEWS_FIELD: review_type
        })

        result = analyze_topics()
        result.update({
            Constants.BOW_TYPE_FIELD: bow_type,
            Constants.TOPIC_MODEL_TARGET_REVIEWS_FIELD: review_type
        })
        results.append(result)

    for result in results:
        print(result)

    prefix = Constants.RESULTS_FOLDER + Constants.ITEM_TYPE + \
        '_topic_model_context_richness'
    csv_file_path = prefix + '.csv'
    json_file_path = prefix + '.json'
    headers = sorted(results[0].keys())
    # ETLUtils.save_csv_file(csv_file_path, results, headers)
    # ETLUtils.save_json_file(json_file_path, results)


def main():
    full_cycle()


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
