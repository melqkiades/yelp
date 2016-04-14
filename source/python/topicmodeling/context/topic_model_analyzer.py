import copy
import json
import os
import time

from pandas import DataFrame

from topicmodeling.context import topic_model_creator
from utils.constants import Constants

context_words = {
    "ball",
    "bf",
    "birthday",
    "boyfriend",
    "breakfast",
    "brunch",
    "car",
    "cars",
    "casual",
    "christmas",
    "college",
    "companion",
    "conversation",
    "coupon",
    "coupons",
    "coworker",
    "dad",
    "date",
    "daughter",
    "deal",
    "deliver",
    "dinner",
    "discount",
    "drive",
    "driven",
    "driving",
    "event",
    "families",
    "father",
    "football",
    "friday",
    "friend",
    "friends",
    "game",
    "games",
    "gf",
    "girlfriend",
    "girls",
    "grandma",
    "group",
    "groupon",
    "happy",
    "hour",
    "husband",
    "inside",
    "karaoke",
    "kid",
    "kids",
    "ladies",
    "lunch",
    "lunch",
    "meeting",
    "men",
    "mom",
    "monday",
    "morning",
    "morning",
    "mother",
    "night",
    "night",
    "nights",
    "outdoor",
    "outside",
    "parking",
    "party",
    "patio",
    "romantic",
    "saturday",
    "school",
    "son",
    "sports",
    "summer",
    "sunday",
    "takeout",
    "thanksgiving",
    "thru",
    "tuesday",
    "tv",
    "tvs",
    "vacation",
    "valet",
    "wednesday",
    "weekday",
    "weekend",
    "wife",
    "women",
    "work",
    "young",
}


def load_topic_model(cycle_index, fold_index):

    print(Constants._properties)

    print('%s: export topic model' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    lda_based_context =\
        topic_model_creator.load_topic_model(cycle_index, fold_index)
    print('%s: loaded topic model' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    # lda_based_context.epsilon = Constants.LDA_EPSILON
    lda_based_context.epsilon = Constants.LDA_EPSILON
    lda_based_context.update_reviews_with_topics()
    lda_based_context.get_context_rich_topics()

    return lda_based_context


def export_topics(cycle_index, fold_index, epsilon):

    new_properties = copy.deepcopy(Constants._properties)
    new_properties['lda_epsilon'] = epsilon
    Constants.update_properties(new_properties)

    lda_based_context = load_topic_model(cycle_index, fold_index)

    file_name = Constants.DATASET_FOLDER + 'all_reviews_topic_model_' + \
        Constants.ITEM_TYPE + '_' + \
        str(Constants.LDA_NUM_TOPICS) + '_' + \
        str(Constants.LDA_MODEL_PASSES) + '_' + \
        str(Constants.LDA_MODEL_ITERATIONS) + '_' + \
        str(Constants.LDA_EPSILON) + \
        '-2.csv'
    print(file_name)

    num_words = 10
    headers = [
        'topic_id',
        'ratio',
        # 'words_ratio',
        # 'past_verbs_ratio',
        # 'frq',
        # 'specific_frq',
        # 'generic_frq',
        # 'log_words',
        # 'specific_log_words',
        # 'generic_log_words',
        # 'log_past_verbs',
        # 'specific_log_past_verbs',
        # 'generic_log_past_verbs'
    ]

    for i in range(num_words):
        headers.append('word' + str(i))

    results = []

    for topic in range(lda_based_context.num_topics):

        # pri

        result = {}
        result['topic_id'] = topic
        result['ratio'] = lda_based_context.topic_ratio_map[topic]
        # result['words_ratio'] = topic_ratio_map[topic]['words_ratio']
        # result['past_verbs_ratio'] = topic_ratio_map[topic]['past_verbs_ratio']
        # result['frq'] = topic_ratio_map[topic]['weighted_frq']['review_frequency']
        # result['specific_frq'] = topic_ratio_map[topic]['specific_weighted_frq']['review_frequency']
        # result['generic_frq'] = topic_ratio_map[topic]['generic_weighted_frq']['review_frequency']
        # result['log_words'] = topic_ratio_map[topic]['weighted_frq']['log_words_frequency']
        # result['specific_log_words'] = topic_ratio_map[topic]['specific_weighted_frq']['log_words_frequency']
        # result['generic_log_words'] = topic_ratio_map[topic]['generic_weighted_frq']['log_words_frequency']
        # result['log_past_verbs'] = topic_ratio_map[topic]['weighted_frq']['log_past_verbs_frequency']
        # result['specific_log_past_verbs'] = topic_ratio_map[topic]['specific_weighted_frq']['log_past_verbs_frequency']
        # result['generic_log_past_verbs'] = topic_ratio_map[topic]['generic_weighted_frq']['log_past_verbs_frequency']
        result.update(split_topic(lda_based_context.topic_model.print_topic(topic, topn=num_words)))

        # print(lda_based_context.topic_model.print_topic(topic, topn=num_words))
        results.append(result)
    analyze_topics(results)
    #
    # ETLUtils.save_csv_file(file_name, results, headers)


def split_topic(topic_string):
    """
    Splits a topic into dictionary containing each word

    :type topic_string: str
    :param topic_string:
    """

    words_dict = {}
    index = 0
    topic_words = topic_string.split(' + ')
    topic_score = 0.0
    for topic_word in topic_words:
        word = topic_word.split('*')[1]
        word_score = float(topic_word.split('*')[0])
        words_dict['word' + str(index)] = topic_word
        if word in context_words:
            topic_score += word_score
        index += 1

    words_dict['score'] = topic_score

    # print(words_dict['score'])

    return words_dict


def analyze_topics(topic_data):

    data_frame = DataFrame.from_dict(topic_data)
    # data_frame.info()
    # print(data_frame.head())
    # print(data_frame)

    scores = {}
    scores['all_ratio_mean_score'] = data_frame['score'].mean()
    scores['high_ratio_mean_score'] = data_frame[data_frame.ratio > 1.0]['score'].mean()
    scores['low_ratio_mean_score'] = data_frame[data_frame.ratio < 1.0]['score'].mean()
    scores['all_ratio_count'] = data_frame[data_frame.score > 0.0]['topic_id'].count()
    scores['high_ratio_count'] = data_frame[(data_frame.ratio > 1.0) & (data_frame.score > 0.0)]['topic_id'].count()
    scores['low_ratio_count'] = data_frame[(data_frame.ratio < 1.0) & (data_frame.score > 0.0)]['topic_id'].count()

    results = copy.deepcopy(Constants._properties)
    results.update(scores)

    write_results_to_json(results)

    print('all mean score: %f' % scores['all_ratio_mean_score'])
    print('greater than 1.0 mean score: %f' % scores['high_ratio_mean_score'])
    print('lower than 1.0 mean score: %f' % scores['low_ratio_mean_score'])
    print('all ratio count: %f' % scores['all_ratio_count'])
    print('greater than 1.0 count: %f' % scores['high_ratio_count'])
    print('lower than 1.0 count: %f' % scores['low_ratio_count'])
    # print('\nN/A')
    # print(data_frame[data_frame.ratio == 'N/A'].mean())
    # print('\n> 1.0')
    # print(data_frame[(data_frame.ratio > 1.0) & (data_frame.ratio != 'N/A')].mean())
    # print('\n< 1.0')
    # print(data_frame[(data_frame.ratio < 1.0) & (data_frame.ratio != 'N/A')].mean())
    # print('\nN/A')
    # print(data_frame[data_frame.ratio == 'N/A'].mean())


def write_results_to_json(results):
    file_name = Constants.DATASET_FOLDER + 'topic_model_analysis_' + \
                Constants.ITEM_TYPE + \
                '.json'
    print(file_name)

    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            json.dump(results, f)
            f.write('\n')
    else:
        with open(file_name, 'a') as f:
            json.dump(results, f)
            f.write('\n')


start = time.time()

epsilon_list = [0.04, 0.08, 0.12]
for epsilon in epsilon_list:
    export_topics(0, 0, epsilon)
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
