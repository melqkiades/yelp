import copy
import time

import itertools
import xlsxwriter
from pandas import DataFrame

from etl import ETLUtils
from topicmodeling.context import topic_model_creator
from utils.constants import Constants


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
    'gambling': {'gamble', 'casino', 'slot', 'roulette', '@', '@@'},
    'party': {'party', 'friend', 'music', 'group', 'nightlife', 'dj'},
    'business': {'business', 'work', 'job', 'colleague', 'coworker', '@'},
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

hotel_context_words = set()
for groups in grouped_hotel_context_words.values():
    hotel_context_words |= groups

restaurant_context_words = set()
for groups in grouped_restaurant_context_words.values():
    restaurant_context_words |= groups

all_context_words = {
    'hotel': hotel_context_words,
    'restaurant': restaurant_context_words
}



def load_topic_model(cycle_index, fold_index):

    print(Constants._properties)

    print('%s: get_records_to_predict_topn topic model' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    lda_based_context =\
        topic_model_creator.load_topic_model(cycle_index, fold_index)
    print('%s: loaded topic model' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    # lda_based_context.epsilon = Constants.LDA_EPSILON
    lda_based_context.epsilon = Constants.LDA_EPSILON
    lda_based_context.update_reviews_with_topics()
    lda_based_context.get_context_rich_topics()
    print('epsilon', lda_based_context.epsilon)

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
        'weighted_frequency',
        'score',
        'combined_scores'
    ]

    for i in range(num_words):
        headers.append('word' + str(i))

    results = []

    topic_ratio_map = lda_based_context.topic_ratio_map
    print(topic_ratio_map)

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
        result['weighted_frequency'] =\
            lda_based_context.topic_weighted_frequency_map[topic]
        result.update(split_topic(
            lda_based_context.topic_model.print_topic(topic, topn=num_words)))
        results.append(result)
    topic_model_score = analyze_topics(results, lda_based_context)

    generate_excel_file(results)
    # ETLUtils.save_csv_file(file_name, results, headers)

    return topic_model_score


def split_topic(topic_string):
    """
    Splits a topic into dictionary containing each word

    :type topic_string: str
    :param topic_string:
    """

    context_words = []
    if Constants.ITEM_TYPE == 'hotel':
        for values in grouped_hotel_context_words.values():
            context_words.extend(values)
    elif Constants.ITEM_TYPE == 'restaurant':
        for values in grouped_restaurant_context_words.values():
            context_words.extend(values)

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

    context_topic_threshold = 0.1

    scores = {}
    num_topics = Constants.LDA_NUM_TOPICS
    scores['num_topics'] = num_topics
    all_ratio_mean_score = data_frame[
        data_frame.weighted_frequency > Constants.LDA_ALPHA]['score'].mean()
    scores['all_ratio_mean_score'] = all_ratio_mean_score
    high_ratio_mean_score = data_frame[(data_frame.ratio > 1.0) & (data_frame.weighted_frequency > Constants.LDA_ALPHA)]['score'].mean()
    low_ratio_mean_score = data_frame[(data_frame.ratio < 1.0) & (data_frame.weighted_frequency > Constants.LDA_ALPHA)]['score'].mean()
    # scores['all_ratio_count'] = data_frame[data_frame.score > 0.1]['topic_id'].count()
    high_ratio_count = data_frame[
        (data_frame.ratio > 1.0) & (data_frame.score > context_topic_threshold) & (data_frame.weighted_frequency > Constants.LDA_ALPHA)]['topic_id'].count()
    low_ratio_count = data_frame[
        (data_frame.ratio < 1.0) & (data_frame.score > context_topic_threshold) & (data_frame.weighted_frequency > Constants.LDA_ALPHA)]['topic_id'].count()
    num_context_topics = len(lda_based_context.context_rich_topics)
    scores['num_context_topics'] = num_context_topics
    scores['document_level'] = Constants.DOCUMENT_LEVEL
    scores['topic_weighting_method'] = Constants.TOPIC_WEIGHTING_METHOD
    scores['alpha'] = Constants.LDA_ALPHA
    scores['epsilon'] = Constants.LDA_EPSILON

    high_ratio_count2 = float(
        data_frame[(data_frame.ratio > 1.0) & (data_frame.weighted_frequency > Constants.LDA_ALPHA)]['score'].count())
    low_ratio_count2 = float(data_frame[(data_frame.ratio < 1.0) & (data_frame.weighted_frequency > Constants.LDA_ALPHA)]['score'].count())

    weighted_high_ratio_count = float(high_ratio_count / high_ratio_count2)
    weighted_low_ratio_count = float(low_ratio_count / low_ratio_count2)
    weighted_ratio_count =\
        (weighted_high_ratio_count / weighted_low_ratio_count)\
        if weighted_low_ratio_count != 0\
        else 'N/A'
    scores['weighted_ratio_count'] = weighted_ratio_count
    score_ratio =\
        (high_ratio_mean_score / low_ratio_mean_score)\
        if low_ratio_mean_score != 0\
        else 'N/A'
    scores['score_ratio'] = score_ratio
    scores['combined_score'] =\
        (all_ratio_mean_score * score_ratio * num_context_topics / num_topics)\
        if all_ratio_mean_score != 'N/A' and score_ratio != 'N/A'\
        else 'N/A'

    results = copy.deepcopy(Constants._properties)
    results.update(scores)

    print('all mean score: %f' % scores['all_ratio_mean_score'])
    # print('greater than 1.0 mean score: %f' % scores['high_ratio_mean_score'])
    # print('lower than 1.0 mean score: %f' % scores['low_ratio_mean_score'])
    # print('all ratio count: %f' % scores['all_ratio_count'])
    # print('greater than 1.0 count: %f' % scores['high_ratio_count'])
    # print('lower than 1.0 count: %f' % scores['low_ratio_count'])
    # print('weighted greater than 1.0 count: %f' % scores['weighted_high_ratio_count'])
    # print('weighted lower than 1.0 count: %f' % scores['weighted_low_ratio_count'])
    print('weighted ratio:', scores['weighted_ratio_count'])
    print('score ratio:', scores['score_ratio'])
    print('combined score:', scores['combined_score'])

    # write_results(results)

    return scores


def write_results(results):
    json_file_name = Constants.DATASET_FOLDER + 'topic_model_analysis_' + \
                Constants.ITEM_TYPE + \
                '.json'
    csv_file_name = Constants.DATASET_FOLDER + 'topic_model_analysis_' + \
                Constants.ITEM_TYPE + \
                '.csv'
    print(json_file_name)
    print(csv_file_name)

    # if not os.path.exists(json_file_name):
    #     with open(json_file_name, 'w') as f:
    #         json.dump(results, f)
    #         f.write('\n')
    # else:
    #     with open(json_file_name, 'a') as f:
    #         json.dump(results, f)
    #         f.write('\n')


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


def generate_excel_file(records):
    my_context_words = []
    if Constants.ITEM_TYPE == 'hotel':
        for values in grouped_hotel_context_words.values():
            my_context_words.extend(values)
    elif Constants.ITEM_TYPE == 'restaurant':
        for values in grouped_restaurant_context_words.values():
            my_context_words.extend(values)

    workbook = xlsxwriter.Workbook('/Users/fpena/tmp/conditional_format-5.xlsx')
    worksheet7 = workbook.add_worksheet()

    yellow_format = workbook.add_format()
    yellow_format.set_pattern(1)  # This is optional when using a solid fill.
    yellow_format.set_bg_color('yellow')

    cyan_format = workbook.add_format()
    cyan_format.set_pattern(1)  # This is optional when using a solid fill.
    cyan_format.set_bg_color('cyan')

    green_format = workbook.add_format()
    green_format.set_pattern(1)  # This is optional when using a solid fill.
    green_format.set_bg_color('green')

    num_words = 10
    headers = [
        'topic_id',
        'ratio',
        'score',
        'weighted_frequency'
    ]
    for i in range(num_words):
        headers.append('word' + str(i))

    data = [[record[column] for column in headers] for record in records]
    headers = [{'header': header} for header in headers]
    num_topics = Constants.LDA_NUM_TOPICS
    print(data)
    print(headers)

    # worksheet7.add_table('B2:N52', {'data': data, 'columns': headers})

    for row_index, row_data in enumerate(data):
        for column_index, cell_value in enumerate(row_data[:4]):
            worksheet7.write(row_index + 2, column_index + 1, cell_value)

    # Add words
    for row_index, row_data in enumerate(data):
        for column_index, cell_value in enumerate(row_data[4:]):
            word = cell_value.split('*')[1]
            if word in my_context_words:
                worksheet7.write(row_index + 2, column_index + 5, cell_value.decode('utf-8'), cyan_format)
            else:
                worksheet7.write(row_index + 2, column_index + 5, cell_value.decode('utf-8'))

    worksheet7.conditional_format(2, 3, num_topics + 1, 3, {'type': 'cell',
                                             'criteria': '>=',
                                             'value': 0.1,
                                             'format': yellow_format})

    worksheet7.add_table(1, 1, num_topics + 1, 14, {'columns': headers})
    # worksheet7.add_table('B2:N302', {'columns': headers})

    # Set widths
    worksheet7.set_column(1, 1, 7)
    worksheet7.set_column(3, 3, 7)
    worksheet7.set_column(4, 4, 8)
    worksheet7.set_column(5, 15, 14)
    workbook.close()


def main():
    # export_topics(0, 0)
    topic_model_scores = []
    epsilon_list = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.35, 0.5]
    # epsilon_list = [0.01]
    alpha_list = [0.0]
    num_topics_list = [30, 50, 75, 100, 150, 300]
    # # num_topics_list = [150]
    # document_level_list = ['review', 'sentence', 1]
    document_level_list = [1]
    # topic_weighting_methods = ['binary', 'probability']
    topic_weighting_methods = ['probability']
    num_cycles = len(epsilon_list) * len(alpha_list) * len(num_topics_list) *\
                 len(document_level_list) * len(topic_weighting_methods)
    cycle_index = 1
    for epsilon, alpha, num_topics, document_level, topic_weighting_method in itertools.product(
            epsilon_list, alpha_list, num_topics_list, document_level_list, topic_weighting_methods):
        print('\ncycle_index: %d/%d' % (cycle_index, num_cycles))
        new_dict = {
            'lda_num_topics': num_topics,
            'document_level': document_level,
            'topic_weighting_method': topic_weighting_method,
            'lda_alpha': alpha,
            'lda_epsilon': epsilon
        }
        Constants.update_properties(new_dict)
        topic_model_score = export_topics(0, 0)
        topic_model_scores.append(topic_model_score)
        cycle_index += 1
    #
    csv_file_name = Constants.DATASET_FOLDER + 'topic_model_analysis_' + \
                    Constants.ITEM_TYPE + \
                    '.csv'
    print(csv_file_name)
    ETLUtils.save_csv_file(csv_file_name, topic_model_scores, topic_model_scores[0].keys())


# start = time.time()
# main()
# export_topics(0, 0)
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))

# generate_excel_file()

