import copy
import json
import os
import time

import itertools

import nltk
import numpy
from pandas import DataFrame

from etl import ETLUtils
from topicmodeling.context import topic_model_creator
from utils.constants import Constants

restaurant_context_words = {
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

hotel_context_words = {
    "airport",
    "anniversary",
    "attended",
    "autumn",
    "bike",
    "boyfriend",
    "bus",
    "business",
    "car",
    "carts",
    "children",
    "christmas",
    "colleagues",
    "conference",
    "course",
    "crowd",
    "date",
    "dog",
    "dogs",
    "driving",
    "engagement",
    "facial",
    "family",
    "fiance",
    "fiancee",
    "fitness",
    "flight",
    "football",
    "friday",
    "friend",
    "friends",
    "game",
    "games",
    "girlfriend",
    "golf",
    "grandfather",
    "grandma",
    "grandmother",
    "grandpa",
    "grandparents",
    "group",
    "groupon",
    "honeymoon",
    "horse",
    "hubby",
    "husband",
    "kids",
    "lazy",
    "marriage",
    "married",
    "massage",
    "mom",
    "monday",
    "music",
    "parents",
    "parking",
    "party",
    "partying",
    "pet",
    "relax",
    "relaxed",
    "relaxing",
    "reservation",
    "rest",
    "romance",
    "romantic",
    "saturday",
    "shuttle",
    "spring",
    "staycation",
    "steam",
    "summer",
    "sunday",
    "tennis",
    "thanksgiving",
    "thursday",
    "tournament",
    "training",
    "transportation",
    "treatment",
    "trip",
    "tuesday",
    "vacation",
    "valet",
    "wedding",
    "wednesday",
    "weekday",
    "weekend",
    "wife",
    "winter",
    "work"
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


def export_topics(cycle_index, fold_index, epsilon=None, alpha=None):

    new_properties = copy.deepcopy(Constants._properties)
    if epsilon is not None:
        new_properties['lda_epsilon'] = epsilon
    if alpha is not None:
        new_properties['lda_alpha'] = alpha

    Constants.update_properties(new_properties)

    lda_based_context = load_topic_model(cycle_index, fold_index)

    file_name = Constants.DATASET_FOLDER + 'all_reviews_topic_model_' + \
        Constants.ITEM_TYPE + '_' + \
        str(Constants.LDA_NUM_TOPICS) + '_' + \
        str(Constants.LDA_MODEL_PASSES) + '_' + \
        str(Constants.LDA_MODEL_ITERATIONS) + '_' + \
        str(Constants.LDA_EPSILON) + \
        '-nouns-complete.csv'
    print(file_name)

    num_words = 10
    headers = [
        'topic_id',
        'ratio',
        'score',
        'words_ratio',
        'past_verbs_ratio',
        'frq',
        'specific_frq',
        'generic_frq',
        'log_words',
        'specific_log_words',
        'generic_log_words',
        'log_past_verbs',
        'specific_log_past_verbs',
        'generic_log_past_verbs'
    ]

    for i in range(num_words):
        headers.append('word' + str(i))

    results = []

    topic_statistics_map = lda_based_context.topic_statistics_map

    num_reviews = len(lda_based_context.records)
    num_specific_reviews = len(lda_based_context.specific_reviews)
    num_generic_reviews = len(lda_based_context.generic_reviews)
    print('num reviews: %d' % num_reviews)
    print('num specific reviews: %d' % num_specific_reviews)
    print('num generic reviews: %d' % num_generic_reviews)
    print('specific reviews percentage : %f %%' % (float(num_specific_reviews) / num_reviews * 100))
    print('generic reviews percentage : %f %%' % (float(num_generic_reviews) / num_reviews * 100))
    print('number of contextual topics: %d' % len(lda_based_context.context_rich_topics))

    for topic in topic_statistics_map.keys():

        # pri

        result = {}
        result['topic_id'] = topic
        result['ratio'] = topic_statistics_map[topic]['frequency_ratio']
        result['words_ratio'] = topic_statistics_map[topic]['words_ratio']
        result['past_verbs_ratio'] = topic_statistics_map[topic]['past_verbs_ratio']
        result['frq'] = topic_statistics_map[topic]['weighted_frq']['review_frequency']
        result['specific_frq'] = topic_statistics_map[topic]['specific_weighted_frq']['review_frequency']
        result['generic_frq'] = topic_statistics_map[topic]['generic_weighted_frq']['review_frequency']
        result['log_words'] = topic_statistics_map[topic]['weighted_frq']['log_words_frequency']
        result['specific_log_words'] = topic_statistics_map[topic]['specific_weighted_frq']['log_words_frequency']
        result['generic_log_words'] = topic_statistics_map[topic]['generic_weighted_frq']['log_words_frequency']
        result['log_past_verbs'] = topic_statistics_map[topic]['weighted_frq']['log_past_verbs_frequency']
        result['specific_log_past_verbs'] = topic_statistics_map[topic]['specific_weighted_frq']['log_past_verbs_frequency']
        result['generic_log_past_verbs'] = topic_statistics_map[topic]['generic_weighted_frq']['log_past_verbs_frequency']
        result.update(split_topic(lda_based_context.topic_model.print_topic(topic, topn=num_words)))

        # print(lda_based_context.topic_model.print_topic(topic, topn=num_words))
        results.append(result)
    analyze_topics(results)
    #
    ETLUtils.save_csv_file(file_name, results, headers)


def split_topic(topic_string):
    """
    Splits a topic into dictionary containing each word

    :type topic_string: str
    :param topic_string:
    """

    context_words = None
    if Constants.ITEM_TYPE == 'hotel':
        context_words = hotel_context_words
    if Constants.ITEM_TYPE == 'restaurant':
        context_words = restaurant_context_words

    words_dict = {}
    index = 0
    topic_words = topic_string.split(' + ')
    topic_score = 0.0
    for topic_word in topic_words:
        word = topic_word.split('*')[1]
        word_score = float(topic_word.split('*')[0])
        words_dict['word' + str(index)] = topic_word.encode('utf-8')
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

    high_ratio_count = float(data_frame[data_frame.ratio > 1.0]['score'].count())
    low_ratio_count = float(data_frame[data_frame.ratio < 1.0]['score'].count())

    results = copy.deepcopy(Constants._properties)
    results.update(scores)

    write_results_to_json(results)

    print('all mean score: %f' % scores['all_ratio_mean_score'])
    print('greater than 1.0 mean score: %f' % scores['high_ratio_mean_score'])
    print('lower than 1.0 mean score: %f' % scores['low_ratio_mean_score'])
    print('all ratio count: %f' % scores['all_ratio_count'])
    print('greater than 1.0 count: %f' % scores['high_ratio_count'])
    print('lower than 1.0 count: %f' % scores['low_ratio_count'])
    print('weighted greater than 1.0 count: %f' % (float(scores['high_ratio_count'] / high_ratio_count)))
    print('weighted lower than 1.0 count: %f' % (float(scores['low_ratio_count'] / low_ratio_count)))
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


# start = time.time()

# epsilon_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1]
# alpha_list = [0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2]
# num_cycles = len(epsilon_list) * len(alpha_list)
# cycle_index = 1

# export_topics(0, 0)

# # for epsilon, alpha in itertools.product(epsilon_list, alpha_list):
# #     print('cycle_index: %d/%d' % (cycle_index, num_cycles))
# #     export_topics(0, 0, epsilon)
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

tagger = nltk.PerceptronTagger()

time_list = []
accurate_dict = {}
approximate_dict = {'NN': 0, 'JJ': 0, 'VB': 0}

for word in restaurant_context_words:
    cycle_start = time.time()
    tagged_word = tagger.tag([word])[0]
    time_list.append(time.time() - cycle_start)

    word_tag = tagged_word[1]
    if word_tag not in accurate_dict:
        accurate_dict[word_tag] = 0
    accurate_dict[word_tag] += 1

    if word_tag.startswith('NN'):
        approximate_dict['NN'] += 1
    elif word_tag.startswith('JJ'):
        approximate_dict['JJ'] += 1
    elif word_tag.startswith('VB'):
        approximate_dict['VB'] += 1
    else:
        if word_tag not in approximate_dict:
            approximate_dict[word_tag] = 0
        approximate_dict[word_tag] += 1

# print(accurate_dict)
# print(approximate_dict)
#
# print('average cycle time: %f' % numpy.mean(time_list))

