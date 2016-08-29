import csv
import json
import os
import time

import itertools
import xlsxwriter
from pandas import DataFrame

from etl import ETLUtils
from topicmodeling.context import topic_model_creator
from topicmodeling.context.lda_based_context import LdaBasedContext
from topicmodeling.context.nmf_context_extractor import NmfContextExtractor
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

base_file_name = Constants.DATASET_FOLDER + 'topic_model_analysis_' + \
                Constants.ITEM_TYPE
csv_file_name = base_file_name + '.csv'
json_file_name = base_file_name + '.json'

num_words = 10

print(csv_file_name)


def export_topics():

    topic_model_creator.plant_seeds()

    start_time = time.time()

    # lda_based_context = load_topic_model(cycle_index, fold_index)
    records = ETLUtils.load_json_file(Constants.FULL_PROCESSED_RECORDS_FILE)
    lda_based_context = LdaBasedContext(records)
    lda_based_context.generate_review_corpus()
    lda_based_context.build_topic_model()
    lda_based_context.update_reviews_with_topics()
    lda_based_context.get_context_rich_topics()

    file_name = Constants.DATASET_FOLDER + 'all_reviews_topic_model_' + \
        Constants.ITEM_TYPE + '_' + \
        str(Constants.LDA_NUM_TOPICS) + '_' + \
        str(Constants.LDA_MODEL_PASSES) + '_' + \
        str(Constants.LDA_MODEL_ITERATIONS) + '_' + \
        str(Constants.LDA_EPSILON) + \
        '-nouns-complete.csv'
    print(file_name)

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
    print('specific reviews percentage : %f %%' %
          (float(num_specific_reviews) / num_reviews * 100))
    print('generic reviews percentage : %f %%' %
          (float(num_generic_reviews) / num_reviews * 100))
    print('number of contextual topics: %d' %
          len(lda_based_context.context_rich_topics))

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

    end_time = time.time()
    cycle_time = end_time - start_time
    topic_model_score['cycle_time'] = cycle_time

    print("Cycle time = %f seconds" % cycle_time)

    generate_excel_file(results)
    # ETLUtils.save_csv_file(file_name, results, headers)

    return topic_model_score


def export_nmf_topics():

    topic_model_creator.plant_seeds()

    start_time = time.time()

    results = []

    records = ETLUtils.load_json_file(Constants.FULL_PROCESSED_RECORDS_FILE)

    print('num_reviews', len(records))
    # lda_context_utils.discover_topics(my_reviews, 150)
    context_extractor = NmfContextExtractor(records)
    context_extractor.num_topics = Constants.LDA_NUM_TOPICS
    context_extractor.generate_review_bows()
    context_extractor.build_document_term_matrix()
    context_extractor.build_stable_topic_model()
    context_extractor.update_reviews_with_topics()
    context_extractor.get_context_rich_topics()
    topic_model_strings = context_extractor.print_topic_model()
    topic_ratio_map = context_extractor.topic_ratio_map

    for topic in range(Constants.LDA_NUM_TOPICS):
        result = {}
        result['topic_id'] = topic
        result.update(split_topic(topic_model_strings[topic]))
        result['ratio'] = topic_ratio_map[topic]
        result['weighted_frequency'] = \
            context_extractor.topic_weighted_frequency_map[topic]
        results.append(result)

    data_frame = DataFrame.from_dict(results)

    scores = {}
    scores['num_topics'] = Constants.LDA_NUM_TOPICS
    topic_model_score = data_frame['score'].mean()
    scores['topic_model_score'] = topic_model_score
    high_ratio_mean_score = data_frame[(data_frame.ratio > 1.0)]['score'].mean()
    low_ratio_mean_score = data_frame[(data_frame.ratio < 1.0)]['score'].mean()
    scores['git_revision_hash'] = Constants.GIT_REVISION_HASH
    scores['topic_model_type'] = Constants.TOPIC_MODEL_TYPE
    scores['lda_model_passes'] = Constants.LDA_MODEL_PASSES
    scores['lda_model_iterations'] = Constants.LDA_MODEL_ITERATIONS
    scores['document_level'] = Constants.DOCUMENT_LEVEL
    scores['topic_weighting_method'] = Constants.TOPIC_WEIGHTING_METHOD

    separation_score = \
        (high_ratio_mean_score / low_ratio_mean_score) \
        if low_ratio_mean_score != 0 \
        else 'N/A'
    scores['separation_score'] = separation_score
    scores['combined_score'] = \
        (topic_model_score * separation_score) \
        if topic_model_score != 'N/A' and separation_score != 'N/A' \
        else 'N/A'

    print('topic model score: %f' % scores['topic_model_score'])
    print('separation score:', scores['separation_score'])
    print('combined score:', scores['combined_score'])

    print('num topics: %d' % Constants.LDA_NUM_TOPICS)
    print('topic model score: %f' % topic_model_score)

    end_time = time.time()
    cycle_time = end_time - start_time
    scores['cycle_time'] = cycle_time

    print("Cycle time = %f seconds" % cycle_time)

    generate_excel_file(results)
    # ETLUtils.save_csv_file(file_name, results, headers)

    return scores


def split_topic(topic_string):
    """
    Splits a topic into dictionary containing each word

    :type topic_string: str
    :param topic_string:
    """

    my_context_words = []
    if Constants.ITEM_TYPE == 'hotel':
        for values in grouped_hotel_context_words.values():
            my_context_words.extend(values)
    elif Constants.ITEM_TYPE == 'restaurant':
        for values in grouped_restaurant_context_words.values():
            my_context_words.extend(values)

    words_dict = {}
    index = 0
    topic_words = topic_string.split(' + ')
    topic_score = 0.0
    for topic_word in topic_words:
        word = topic_word.split('*')[1]
        word_score = float(topic_word.split('*')[0])
        words_dict['word' + str(index)] = topic_word.encode('utf-8')
        if word in my_context_words:
            topic_score += word_score
        index += 1

    words_dict['score'] = topic_score

    # print(words_dict['score'])

    return words_dict


def analyze_topics(topic_data, lda_based_context):

    data_frame = DataFrame.from_dict(topic_data)

    scores = {}
    num_topics = Constants.LDA_NUM_TOPICS
    scores['num_topics'] = num_topics
    topic_model_score = data_frame[
        data_frame.weighted_frequency > Constants.LDA_ALPHA]['score'].mean()
    scores['topic_model_score'] = topic_model_score
    high_ratio_mean_score = data_frame[
        (data_frame.ratio > 1.0) &
        (data_frame.weighted_frequency > Constants.LDA_ALPHA)]['score'].mean()
    low_ratio_mean_score = data_frame[
        (data_frame.ratio < 1.0) &
        (data_frame.weighted_frequency > Constants.LDA_ALPHA)]['score'].mean()
    # scores['all_ratio_count'] =
    #     data_frame[data_frame.score > 0.1]['topic_id'].count()
    # num_context_topics = len(lda_based_context.context_rich_topics)
    # scores['num_context_topics'] = num_context_topics
    scores['document_level'] = Constants.DOCUMENT_LEVEL
    scores['topic_weighting_method'] = Constants.TOPIC_WEIGHTING_METHOD
    # scores['alpha'] = Constants.LDA_ALPHA
    # scores['epsilon'] = Constants.LDA_EPSILON
    # scores['lda_review_type'] = Constants.LDA_REVIEW_TYPE
    scores['lda_model_passes'] = Constants.LDA_MODEL_PASSES
    scores['lda_model_iterations'] = Constants.LDA_MODEL_ITERATIONS
    scores['git_revision_hash'] = Constants.GIT_REVISION_HASH
    scores['topic_model_type'] = Constants.TOPIC_MODEL_TYPE

    separation_score =\
        (high_ratio_mean_score / low_ratio_mean_score)\
        if low_ratio_mean_score != 0\
        else 'N/A'
    scores['separation_score'] = separation_score
    scores['combined_score'] =\
        (topic_model_score * separation_score)\
        if topic_model_score != 'N/A' and separation_score != 'N/A'\
        else 'N/A'

    results = Constants.get_properties_copy()
    results.update(scores)

    print('topic model score: %f' % scores['topic_model_score'])
    print('separation score:', scores['separation_score'])
    print('combined score:', scores['combined_score'])

    # write_results(results)

    return scores


def generate_excel_file(records):
    my_context_words = []
    if Constants.ITEM_TYPE == 'hotel':
        for values in grouped_hotel_context_words.values():
            my_context_words.extend(values)
    elif Constants.ITEM_TYPE == 'restaurant':
        for values in grouped_restaurant_context_words.values():
            my_context_words.extend(values)

    file_name = Constants.DATASET_FOLDER + 'topic_model_' +\
        Constants.TOPIC_MODEL_TYPE + '_' +\
        Constants.ITEM_TYPE + '_' + \
        str(Constants.LDA_NUM_TOPICS) + 't_' +\
        str(Constants.LDA_MODEL_PASSES) + 'p_' + \
        str(Constants.LDA_MODEL_ITERATIONS) + 'i_' + \
        '.xlsx'
    workbook = xlsxwriter.Workbook(file_name)
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
    # print(data)
    # print(headers)

    # worksheet7.add_table('B2:N52', {'data': data, 'columns': headers})

    for row_index, row_data in enumerate(data):
        for column_index, cell_value in enumerate(row_data[:4]):
            worksheet7.write(row_index + 2, column_index + 1, cell_value)

    # Add words
    for row_index, row_data in enumerate(data):
        for column_index, cell_value in enumerate(row_data[4:]):
            word = cell_value.split('*')[1]
            if word in my_context_words:
                worksheet7.write(
                    row_index + 2, column_index + 5,
                    cell_value.decode('utf-8'), cyan_format
                )
            else:
                worksheet7.write(
                    row_index + 2, column_index + 5, cell_value.decode('utf-8'))

    worksheet7.conditional_format(2, 3, num_topics + 1, 3, {
        'type': 'cell',
        'criteria': '>=',
        'value': 0.1,
        'format': yellow_format})

    worksheet7.add_table(
        1, 1, num_topics + 1, 4 + num_words, {'columns': headers})
    # worksheet7.add_table('B2:N302', {'columns': headers})

    # Set widths
    worksheet7.set_column(1, 1, 7)
    worksheet7.set_column(3, 3, 7)
    worksheet7.set_column(4, 4, 8)
    worksheet7.set_column(5, 15, 14)
    workbook.close()


def main():

    # export_topics(0, 0)
    # epsilon_list = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.35, 0.5]
    epsilon_list = [0.05]
    alpha_list = [0.0]
    # num_topics_list =\
    #     [5, 10, 35, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    num_topics_list = [10, 20, 30, 50, 75, 100, 150, 300]
    # num_topics_list = [150, 300]
    # num_topics_list = [300]
    # document_level_list = ['review', 'sentence', 1]
    document_level_list = [1]
    # topic_weighting_methods = ['binary', 'probability']
    topic_weighting_methods = ['probability']
    # review_type_list = ['specific', 'generic', 'all_reviews']
    review_type_list = ['specific']
    # lda_passes_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    lda_passes_list = [1, 10, 100]
    # lda_passes_list = [1]
    # lda_iterations_list = [50, 100, 200, 400, 800, 2000]
    lda_iterations_list = [50, 100, 200, 500]
    # lda_iterations_list = [50]
    # topic_model_type_list = ['lda']
    topic_model_type_list = ['nmf', 'lda']
    num_cycles = len(epsilon_list) * len(alpha_list) * len(num_topics_list) *\
        len(document_level_list) * len(topic_weighting_methods) *\
        len(review_type_list) * len(lda_passes_list) *\
        len(lda_iterations_list) * len(topic_model_type_list)
    cycle_index = 1
    for epsilon, alpha, num_topics, document_level, topic_weighting_method,\
        review_type, lda_passes, lda_iterations,\
        topic_model_type in itertools.product(
            epsilon_list, alpha_list, num_topics_list, document_level_list,
            topic_weighting_methods, review_type_list, lda_passes_list,
            lda_iterations_list, topic_model_type_list):
        print('\ncycle_index: %d/%d' % (cycle_index, num_cycles))
        new_dict = {
            'lda_num_topics': num_topics,
            'document_level': document_level,
            'topic_weighting_method': topic_weighting_method,
            'lda_alpha': alpha,
            'lda_epsilon': epsilon,
            'lda_review_type': review_type,
            'lda_model_passes': lda_passes,
            'lda_model_iterations': lda_iterations,
            'topic_model_type': topic_model_type
        }

        # print(new_dict)

        Constants.update_properties(new_dict)
        if Constants.TOPIC_MODEL_TYPE == 'lda':
            topic_model_score = export_topics()
        elif Constants.TOPIC_MODEL_TYPE == 'nmf':
            topic_model_score = export_nmf_topics()

        write_results_to_csv(csv_file_name, topic_model_score)
        write_results_to_json(json_file_name, topic_model_score)

        cycle_index += 1


def write_results_to_csv(file_name, results):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            w = csv.DictWriter(f, results.keys())
            w.writeheader()
            w.writerow(results)
    else:
        with open(file_name, 'a') as f:
            w = csv.DictWriter(f, results.keys())
            w.writerow(results)


def write_results_to_json(file_name, results):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            json.dump(results, f)
            f.write('\n')
    else:
        with open(file_name, 'a') as f:
            json.dump(results, f)
            f.write('\n')

# start = time.time()
# main()
# export_topics()
# export_nmf_topics()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

# generate_excel_file()
