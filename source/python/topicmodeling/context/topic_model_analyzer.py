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

single_restaurant_context_words = {
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

single_hotel_context_words = {
    "airport",
    "anniversary",
    "april",
    "attended",
    "august",
    "autumn",
    "bike",
    "birthday"
    "boyfriend",
    "bus",
    "business",
    "car",
    "carts",
    "children",
    "christmas",
    "colleagues",
    "conference",
    "convention",
    "course",
    "crowd",
    "date",
    "december",
    "dog",
    "dogs",
    "driving",
    "engagement",
    "facial",
    "family",
    "february",
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
    "january",
    "june",
    "july",
    "kids",
    "lazy",
    "march",
    "marriage",
    "married",
    "massage",
    "may",
    "mom",
    "monday",
    "music",
    "nightlife",
    "november",
    "october",
    "parents",
    "parking",
    "party",
    "partying",
    "pet",
    "pets",
    "relax",
    "relaxed",
    "relaxing",
    "reservation",
    "rest",
    "romance",
    "romantic",
    "saturday",
    "september",
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


grouped_hotel_context_words = {
    'airport': {'airport', 'shuttle', 'plane', 'flight', 'transportation',
                'bus'},
    'holiday': {'holiday', 'vacation', 'staycation', 'getaway', '@', '@@'},
    'conference': {'conference', 'convention', 'group', 'meeting', 'attended',
                   '@'},
    'pets': {'dog', 'pet', 'cat', '@', '@@', '@@@'},
    'discount': {'discount', 'hotwire', 'groupon', 'deal', 'priceline', '@'},
    'wedding': {'wedding', 'reception', 'ceremony', 'marriage', '@', '@@'},
    'festivities': {'christmas', 'thanksgiving', 'holiday', '@', '@@', '@@@'},
    'family': {'mom', 'dad', 'father', 'mother', 'grandma', 'grandmother',
               'grandpa', 'grandfather', 'parent', 'grandparent',
               'daughter', 'uncle', 'sister', 'brother', 'aunt', 'sibling',
               'child',  'daughter', 'son', 'kid', 'boy', 'girl', 'family'},
    'romantic': {'date', 'anniversary', 'romantic', 'girlfriend',
                 'boyfriend', 'bf', 'gf', 'hubby', 'husband', 'wife',
                 'fiance', 'fiancee', 'weekend', 'getaway', 'romance'},
    'anniversary': {'husband', 'wife', 'weekend', 'anniversary', 'hubby', '@'},
    'gambling': {'gamble', 'casino', 'slot', 'machine', 'roulette', '@'},
    'party': {'party', 'friend', 'music', 'group', 'nightlife', 'dj'},
    'business': {'busines', 'work', 'job', 'colleague', 'coworker', '@'},
    'parking': {'car', 'parking', 'valet', 'driver', '@', '@@'},
    'season': {'winter', 'spring', 'summer', 'fall', 'autumn', '@'},
    'month': {'january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november',
              'december'},
    'day': {'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'weekday', 'weekend'},
    'occasion': {'birthday', 'anniversary', 'celebration', 'date',
                 'wedding', 'honeymoon'},
    'sport_event': {'football', 'baseball', 'basketball', 'game', 'field',
                    'match', 'tournament', 'ticket'},
    'outdoor': {'golf', 'tenni', 'court', 'field', 'horse', 'cabana'
                'training', 'exercise', 'bike', 'cycle', 'kart', 'cart',
                'fitness'},
    'relax': {
        'relax', 'quiet', 'getaway', 'stress', 'relief', 'massage',
        'spa', 'steam', 'jacuzzi', 'facial', 'treatment', 'relaxing',
        'treatment'
    },
    'accessibility': {
        'wheelchair', 'handicap', 'ramp', '@', '@@', '@@@'
    },
    # 'non-contextual': {'room'}
}

grouped_restaurant_context_words = {
    'breakfast': {'brunch', 'breakfast', 'morning', 'pancakes', 'omelette',
                  'waffle'},
    'lunch': {'afternoon', 'lunch', 'noon',  '@', '@@', '@@@'},
    'dinner': {'dinner', 'evening', 'night',  '@', '@@', '@@@'},
    'romantic': {'date', 'night', 'anniversary', 'romantic', 'girlfriend',
                 'boyfriend', 'bf', 'gf', 'hubby', 'husband', 'wife',
                 'fiance', 'fiancee', 'weekend'},
    'party': {'party', 'friend', 'music', 'group', 'disco',
              'club', 'guy', 'people', 'night', 'nightlife'},
    'kids': {'child', 'kid', 'boy', 'girl', 'family', '@'},
    'parking': {'parking', 'car', 'valet', 'driver', '@', '@@'},
    'work': {'busines', 'colleague', 'workplace', 'job', 'meeting', 'coworker',
             'office'},
    'family': {'mom', 'dad', 'father', 'mother', 'grandma', 'grandmother',
               'grandpa', 'grandfather', 'parent', 'grandparent',
               'daughter', 'uncle', 'sister', 'brother', 'aunt', 'sibling',
               'daughter', 'son'},
    'friends': {'friend', 'group', 'girl', 'boy', 'guy', '@'},
    'time': {'morning', 'noon', 'afternoon', 'evening', 'night', '@'},
    'birthday': {'birthday', 'celebration', 'event', '@', '@@', '@@@'},
    'discount': {'deal', 'coupon', 'groupon', 'discount', '@', '@@'},
    'takeaway': {'delivery', 'takeaway', 'drive', 'thru', 'takeout',
                 'deliver'},
    'sports': {'sports', 'match', 'game', 'tv', 'football',
               'baseball', 'basketball', 'nfl'},
    'karaoke': {'song', 'karaoke', 'music', '@', '@@', '@@@'},
    'outdoor': {'outdoor', 'patio', 'outside', 'summer', '@', '@@'},
    'day': {'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'weekday', 'weekend'},
    # 'season': {'winter', 'summer', 'fall', 'autumn'},
    'accessibility': {'wheelchair', 'handicap', 'ramp', '@', '@@', '@@@'},
    # 'non-contextual': {'food'}
}

context_words = {
    'hotel': grouped_hotel_context_words,
    'restaurant': grouped_restaurant_context_words
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

    topic_model_creator.plant_seeds()

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

    topic_statistics_map = lda_based_context.topic_statistics_map
    topic_ratio_map = lda_based_context.topic_ratio_map

    num_reviews = len(lda_based_context.records)
    num_specific_reviews = len(lda_based_context.specific_reviews)
    num_generic_reviews = len(lda_based_context.generic_reviews)
    print('num reviews: %d' % num_reviews)
    print('num specific reviews: %d' % num_specific_reviews)
    print('num generic reviews: %d' % num_generic_reviews)
    print('specific reviews percentage : %f %%' % (float(num_specific_reviews) / num_reviews * 100))
    print('generic reviews percentage : %f %%' % (float(num_generic_reviews) / num_reviews * 100))
    print('number of contextual topics: %d' % len(lda_based_context.context_rich_topics))

    for topic in topic_ratio_map.keys():
        result = {}
        result['topic_id'] = topic
        result['ratio'] = topic_ratio_map[topic]
        result.update(split_topic(
            lda_based_context.topic_model.print_topic(topic, topn=num_words)))
        results.append(result)

    # for topic in topic_statistics_map.keys():
    #
    #     # pri
    #
    #     result = {}
    #     result['topic_id'] = topic
    #     result['ratio'] = topic_statistics_map[topic]['frequency_ratio']
    #     result['words_ratio'] = topic_statistics_map[topic]['words_ratio']
    #     result['past_verbs_ratio'] = topic_statistics_map[topic]['past_verbs_ratio']
    #     result['frq'] = topic_statistics_map[topic]['weighted_frq']['review_frequency']
    #     result['specific_frq'] = topic_statistics_map[topic]['specific_weighted_frq']['review_frequency']
    #     result['generic_frq'] = topic_statistics_map[topic]['generic_weighted_frq']['review_frequency']
    #     result['log_words'] = topic_statistics_map[topic]['weighted_frq']['log_words_frequency']
    #     result['specific_log_words'] = topic_statistics_map[topic]['specific_weighted_frq']['log_words_frequency']
    #     result['generic_log_words'] = topic_statistics_map[topic]['generic_weighted_frq']['log_words_frequency']
    #     result['log_past_verbs'] = topic_statistics_map[topic]['weighted_frq']['log_past_verbs_frequency']
    #     result['specific_log_past_verbs'] = topic_statistics_map[topic]['specific_weighted_frq']['log_past_verbs_frequency']
    #     result['generic_log_past_verbs'] = topic_statistics_map[topic]['generic_weighted_frq']['log_past_verbs_frequency']
    #     result.update(split_topic(lda_based_context.topic_model.print_topic(topic, topn=num_words)))
    #
    #     # print(lda_based_context.topic_model.print_topic(topic, topn=num_words))
    #     results.append(result)
    analyze_topics(results, lda_based_context)
    #
    # ETLUtils.save_csv_file(file_name, results, headers)


def split_topic(topic_string):
    """
    Splits a topic into dictionary containing each word

    :type topic_string: str
    :param topic_string:
    """

    context_words = None
    if Constants.ITEM_TYPE == 'hotel':
        context_words = single_hotel_context_words
    if Constants.ITEM_TYPE == 'restaurant':
        context_words = single_restaurant_context_words

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


def analyze_topics(topic_data, lda_based_context):

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
    scores['num_context_topics'] = len(lda_based_context.context_rich_topics)

    high_ratio_count = float(
        data_frame[data_frame.ratio > 1.0]['score'].count())
    low_ratio_count = float(data_frame[data_frame.ratio < 1.0]['score'].count())

    scores['weighted_high_ratio_count'] = float(scores['high_ratio_count'] / high_ratio_count)
    scores['weighted_low_ratio_count'] = float(scores['low_ratio_count'] / low_ratio_count)

    results = copy.deepcopy(Constants._properties)
    results.update(scores)

    # write_results_to_json(results)

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


def analyze_manual_topics():
    records = ETLUtils.load_json_file(Constants.FULL_PROCESSED_RECORDS_FILE)

    context_topics = context_words[Constants.ITEM_TYPE]
    topic_counts = {'all': {}, 'specific': {}, 'generic': {}}
    review_type_counts = {Constants.SPECIFIC: 0.0, Constants.GENERIC: 0.0}
    ratio_counts = {}

    # Init counts
    for review_type in topic_counts:
        for topic in context_topics:
            topic_counts[review_type][topic] = 0.0

    for topic in context_topics:
        ratio_counts[topic] = 0.0

    # my_records = [
    #     {Constants.PREDICTED_CLASS_FIELD: Constants.SPECIFIC,
    #      Constants.BOW_FIELD: ['room']},
    #     {Constants.PREDICTED_CLASS_FIELD: Constants.SPECIFIC,
    #      Constants.BOW_FIELD: ['room']},
    #     {Constants.PREDICTED_CLASS_FIELD: Constants.SPECIFIC,
    #      Constants.BOW_FIELD: ['pool']},
    #     {Constants.PREDICTED_CLASS_FIELD: Constants.GENERIC,
    #      Constants.BOW_FIELD: ['room']},
    #     {Constants.PREDICTED_CLASS_FIELD: Constants.GENERIC,
    #      Constants.BOW_FIELD: ['room']},
    #     {Constants.PREDICTED_CLASS_FIELD: Constants.GENERIC,
    #      Constants.BOW_FIELD: ['room']},
    #     {Constants.PREDICTED_CLASS_FIELD: Constants.GENERIC,
    #      Constants.BOW_FIELD: ['staff']},
    # ]

    index = 0
    for record in records:

        review_type = record[Constants.PREDICTED_CLASS_FIELD]
        review_type_counts[review_type] += 1

        # print('\n***************************************')
        # print(record[Constants.TEXT_FIELD])
        # print(index, record[Constants.PREDICTED_CLASS_FIELD], record[Constants.BOW_FIELD])

        for topic, topic_words in context_topics.items():
            if len(topic_words.intersection(record[Constants.BOW_FIELD])) > 0:
                topic_counts['all'][topic] += 1
                topic_counts[review_type][topic] += 1

        index += 1

    for topic in context_topics:
        specific_topic_count = topic_counts[Constants.SPECIFIC][topic]
        total_specific_count = review_type_counts[Constants.SPECIFIC]
        specific_weighted = specific_topic_count / total_specific_count

        generic_topic_count = topic_counts[Constants.GENERIC][topic]
        total_generic_count = review_type_counts[Constants.GENERIC]
        generic_weighted = generic_topic_count / total_generic_count
        # print('specific_weighted', specific_weighted)
        # print('generic_weighted', generic_weighted)
        ratio_counts[topic] = specific_weighted / generic_weighted

    print(review_type_counts)
    print(topic_counts)
    print(ratio_counts)


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
#
# # epsilon_list = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.35, 0.5]
# epsilon_list = [0.001]
# # alpha_list = [0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2]
# alpha_list = [0.005]
# num_cycles = len(epsilon_list) * len(alpha_list)
# cycle_index = 1
#
# export_topics(0, 0)
#
# for epsilon, alpha in itertools.product(epsilon_list, alpha_list):
#     print('cycle_index: %d/%d' % (cycle_index, num_cycles))
#     export_topics(0, 0, epsilon, alpha)
#     cycle_index += 1
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

tagger = nltk.PerceptronTagger()

time_list = []
accurate_dict = {}
approximate_dict = {'NN': 0, 'JJ': 0, 'VB': 0}

for word in single_restaurant_context_words:
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


def see_topic_analysis_results():
    topic_analysis_file = Constants.DATASET_FOLDER + 'topic_model_analysis_' + \
                          Constants.ITEM_TYPE + '.json'

    results = ETLUtils.load_json_file(topic_analysis_file)

    index = 0
    for result in results:
        score_ratio = result['high_ratio_mean_score'] / result[
            'low_ratio_mean_score']
        count_ratio = result['weighted_high_ratio_count'] / result[
            'weighted_low_ratio_count']
        print(index, score_ratio, count_ratio,
              result['high_ratio_mean_score'],
              result['low_ratio_mean_score'],
              result['lda_epsilon'], result['topic_weighting_method'],
              result['num_context_topics'], result['lda_num_topics'])
        index += 1


# see_topic_analysis_results()

# print(accurate_dict)
# print(approximate_dict)
#
# print('average cycle time: %f' % numpy.mean(time_list))

# start = time.time()
# analyze_manual_topics()
# # load_topic_model(0, 0)
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

