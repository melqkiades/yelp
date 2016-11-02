import csv
import json
import os
import time

import cPickle as pickle
import itertools

import numpy
import xlsxwriter
from pandas import DataFrame

from etl import ETLUtils
from topicmodeling.context import topic_model_creator
from topicmodeling.hungarian import HungarianError
from topicmodeling.jaccard_similarity import AverageJaccard
from topicmodeling.jaccard_similarity import RankingSetAgreement
from utils import utilities
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
    'gambling': {'gamble', 'casino', 'slot', 'poker' 'roulette', '@'},
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
    'yelp_hotel': grouped_hotel_context_words,
    'fourcity_hotel': grouped_hotel_context_words,
    'yelp_restaurant': grouped_restaurant_context_words
}

hotel_context_words = set()
for groups in grouped_hotel_context_words.values():
    hotel_context_words |= groups

restaurant_context_words = set()
for groups in grouped_restaurant_context_words.values():
    restaurant_context_words |= groups

all_context_words = {
    'yelp_hotel': hotel_context_words,
    'fourcity_hotel': hotel_context_words,
    'yelp_restaurant': restaurant_context_words
}


def get_topic_model_terms(context_extractor, num_terms):

    context_extractor.num_topics = Constants.TOPIC_MODEL_NUM_TOPICS
    topic_model_strings = context_extractor.print_topic_model(num_terms)
    topic_term_matrix = []

    for topic in range(Constants.TOPIC_MODEL_NUM_TOPICS):
        terms = topic_model_strings[topic].split(" + ")
        terms = [term.partition("*")[2] for term in terms]
        topic_term_matrix.append(terms)

    return topic_term_matrix


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
    rank_score = 0
    for topic_word in topic_words:
        word = topic_word.split('*')[1]
        word_probability_score = float(topic_word.split('*')[0])
        words_dict['word' + str(index)] = topic_word.encode('utf-8')
        if word in my_context_words:
            probability_score += word_probability_score
            word_rank_score = 5 - index if index < 5 else 0
            rank_score += word_rank_score
        index += 1

    words_dict['probability_score'] = probability_score
    words_dict['rank_score'] = rank_score

    return words_dict


def analyze_topics(include_stability=True):

    start_time = time.time()

    utilities.plant_seeds()

    if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
        records = ETLUtils.load_json_file(
            Constants.TOPIC_MODEL_PROCESSED_RECORDS_FILE)
    else:
        records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)
    print('num_reviews', len(records))

    context_extractor =\
        topic_model_creator.create_topic_model(records, None, None)

    topic_data = []

    for topic in range(Constants.TOPIC_MODEL_NUM_TOPICS):
        result = {}
        result['topic_id'] = topic
        result.update(split_topic(context_extractor.print_topic_model(
            num_terms=Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)[topic]))
        result['ratio'] = context_extractor.topic_ratio_map[topic]
        result['weighted_frequency'] = \
            context_extractor.topic_weighted_frequency_map[topic]
        topic_data.append(result)

    generate_excel_file(topic_data)
    data_frame = DataFrame.from_dict(topic_data)

    scores = {}
    scores['num_topics'] = Constants.TOPIC_MODEL_NUM_TOPICS
    probability_score = data_frame['probability_score'].mean()
    rank_score = data_frame['rank_score'].mean()
    scores['probability_score'] = probability_score
    scores['rank_score'] = rank_score
    high_ratio_mean_score = data_frame[
        (data_frame.ratio > Constants.CONTEXT_EXTRACTOR_BETA)]['probability_score'].mean()
    low_ratio_mean_score = data_frame[
        (data_frame.ratio < Constants.CONTEXT_EXTRACTOR_BETA)]['probability_score'].mean()
    high_rank_ratio_mean_score = data_frame[
        (data_frame.ratio > Constants.CONTEXT_EXTRACTOR_BETA)]['rank_score'].mean()
    low_rank_ratio_mean_score = data_frame[
        (data_frame.ratio < Constants.CONTEXT_EXTRACTOR_BETA)]['rank_score'].mean()

    stability = None
    if include_stability:
        stability = calculate_topic_stability().mean()
    scores['stability'] = stability

    separation_score =\
        (high_ratio_mean_score / low_ratio_mean_score)\
        if low_ratio_mean_score != 0\
        else 'N/A'
    rank_separation_score =\
        (high_rank_ratio_mean_score / low_rank_ratio_mean_score)\
        if low_ratio_mean_score != 0\
        else 'N/A'
    scores['separation_score'] = separation_score
    scores['rank_separation_score'] = rank_separation_score
    scores['combined_score'] =\
        (probability_score * separation_score)\
        if probability_score != 'N/A' and separation_score != 'N/A'\
        else 'N/A'

    print('probability score: %f' % scores['probability_score'])
    print('rank score: %f' % scores['rank_score'])
    print('separation score:', scores['separation_score'])
    print('rank separation score:', scores['rank_separation_score'])
    print('combined score:', scores['combined_score'])

    end_time = time.time()
    cycle_time = end_time - start_time
    scores['cycle_time'] = cycle_time

    print("Cycle time = %f seconds" % cycle_time)

    return scores


def calculate_topic_stability():

    Constants.update_properties({
        Constants.NUMPY_RANDOM_SEED_FIELD: Constants.NUMPY_RANDOM_SEED + 10,
        Constants.RANDOM_SEED_FIELD: Constants.RANDOM_SEED + 10
    })
    utilities.plant_seeds()
    Constants.print_properties()

    if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
        records = ETLUtils.load_json_file(
            Constants.TOPIC_MODEL_PROCESSED_RECORDS_FILE)
    else:
        records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)
    print('num_reviews', len(records))

    all_term_rankings = []

    context_extractor =\
        topic_model_creator.create_topic_model(records, None, None)
    terms_matrix = get_topic_model_terms(
        context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
    all_term_rankings.append(terms_matrix)

    for _ in range(Constants.TOPIC_MODEL_STABILITY_ITERATIONS - 1):
        context_extractor = topic_model_creator.train_context_extractor(records)
        terms_matrix = get_topic_model_terms(
            context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
        all_term_rankings.append(terms_matrix)

    return calculate_stability(all_term_rankings)


def calculate_topic_stability_cross_validation():
    all_term_rankings = []

    for i in range(Constants.CROSS_VALIDATION_NUM_FOLDS):
        context_extractor = topic_model_creator.load_topic_model(0, i)
        terms_matrix = get_topic_model_terms(
            context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
        all_term_rankings.append(terms_matrix)

    return calculate_stability(all_term_rankings)


def calculate_stability(all_term_rankings):

    # First argument was the reference term ranking
    reference_term_ranking = all_term_rankings[0]
    all_term_rankings = all_term_rankings[1:]
    r = len(all_term_rankings)
    print("Loaded %d non-reference term rankings" % r)

    # Perform the evaluation
    metric = AverageJaccard()
    matcher = RankingSetAgreement(metric)
    print("Performing reference comparisons with %s ..." % str(metric))
    all_scores = []
    for i in range(r):
        try:
            score = \
                matcher.similarity(reference_term_ranking,
                                   all_term_rankings[i])
            all_scores.append(score)
        except HungarianError:
            msg = \
                "HungarianError: Unable to find results. Algorithm has failed."
            print(msg)
            all_scores.append(float('nan'))

    # Get overall score across all candidates
    all_scores = numpy.array(all_scores)

    print("Stability=%.4f [%.4f,%.4f]" % (
        numpy.nanmean(all_scores), numpy.nanmin(all_scores),
        numpy.nanmax(all_scores)))

    return all_scores


def generate_excel_file(records):
    my_context_words = []
    if 'hotel' in Constants.ITEM_TYPE:
        for values in grouped_hotel_context_words.values():
            my_context_words.extend(values)
    elif 'restaurant' in Constants.ITEM_TYPE:
        for values in grouped_restaurant_context_words.values():
            my_context_words.extend(values)

    file_name = utilities.generate_file_name(
        'topic_model', 'xlsx', Constants.DATASET_FOLDER, None, None, True)
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
        'probability_score',
        'rank_score',
        'weighted_frequency'
    ]
    num_headers = len(headers)
    for i in range(Constants.TOPIC_MODEL_STABILITY_NUM_TERMS):
        headers.append('word' + str(i))

    data = [[record[column] for column in headers] for record in records]
    headers = [{'header': header} for header in headers]
    num_topics = Constants.TOPIC_MODEL_NUM_TOPICS

    for row_index, row_data in enumerate(data):
        for column_index, cell_value in enumerate(row_data[:num_headers]):
            worksheet7.write(row_index + 2, column_index + 1, cell_value)

    # Add words
    for row_index, row_data in enumerate(data):
        for column_index, cell_value in enumerate(row_data[num_headers:]):
            word = cell_value.split('*')[1]
            if word in my_context_words:
                worksheet7.write(
                    row_index + 2, column_index + num_headers + 1,
                    cell_value.decode('utf-8'), cyan_format
                )
            else:
                worksheet7.write(
                    row_index + 2, column_index + num_headers + 1, cell_value.decode('utf-8'))

    worksheet7.conditional_format(2, 3, num_topics + 1, 3, {
        'type': 'cell',
        'criteria': '>=',
        'value': 0.1,
        'format': yellow_format})

    worksheet7.add_table(
        1, 1, num_topics + 1, num_headers + Constants.TOPIC_MODEL_STABILITY_NUM_TERMS,
        {'columns': headers})

    # Set widths
    worksheet7.set_column(1, 1, 7)
    worksheet7.set_column(3, 3, 7)
    worksheet7.set_column(4, 4, 8)
    worksheet7.set_column(5, 15, 14)
    workbook.close()


def main():
    base_file_name = Constants.DATASET_FOLDER + 'topic_model_analysis_' + \
                     Constants.ITEM_TYPE

    csv_file_name = base_file_name + '.csv'
    json_file_name = base_file_name + '.json'
    print(csv_file_name)

    # export_lda_topics(0, 0)
    # epsilon_list = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.35, 0.5]
    epsilon_list = [0.05]
    alpha_list = [0.0]
    # num_topics_list =\
    #     [5, 10, 35, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    # num_topics_list = [10, 20, 30, 50, 75, 100, 150, 300]
    # num_topics_list = [150, 300]
    num_topics_list = [10, 20, 30]
    bow_type_list = ['NN']
    # document_level_list = ['review', 'sentence', 1]
    document_level_list = [1]
    # topic_weighting_methods = ['binary', 'probability']
    topic_weighting_methods = ['probability']
    # review_type_list = ['specific', 'generic', 'all_reviews']
    review_type_list = ['specific']
    # lda_passes_list = [1, 10, 20, 50, 75, 100, 200, 500]
    # lda_passes_list = [1, 10]
    lda_passes_list = [100]
    # lda_iterations_list = [50, 100, 200, 400, 800, 2000]
    # lda_iterations_list = [50, 100, 200, 500]
    lda_iterations_list = [200]
    # topic_model_type_list = ['lda', 'nmf']
    topic_model_type_list = ['nmf']
    num_cycles = len(epsilon_list) * len(alpha_list) * len(num_topics_list) *\
        len(document_level_list) * len(topic_weighting_methods) *\
        len(review_type_list) * len(lda_passes_list) *\
        len(lda_iterations_list) * len(topic_model_type_list) *\
        len(bow_type_list)
    cycle_index = 1
    for epsilon, alpha, num_topics, document_level, topic_weighting_method,\
        review_type, lda_passes, lda_iterations, topic_model_type,\
        bow_type in itertools.product(
            epsilon_list, alpha_list, num_topics_list, document_level_list,
            topic_weighting_methods, review_type_list, lda_passes_list,
            lda_iterations_list, topic_model_type_list, bow_type_list):
        print('\ncycle_index: %d/%d' % (cycle_index, num_cycles))
        new_dict = {
            Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: num_topics,
            Constants.DOCUMENT_LEVEL_FIELD: document_level,
            Constants.TOPIC_WEIGHTING_METHOD_FIELD: topic_weighting_method,
            Constants.CONTEXT_EXTRACTOR_ALPHA_FIELD: alpha,
            Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD: epsilon,
            Constants.TOPIC_MODEL_REVIEW_TYPE_FIELD: review_type,
            Constants.TOPIC_MODEL_PASSES_FIELD: lda_passes,
            Constants.TOPIC_MODEL_ITERATIONS_FIELD: lda_iterations,
            Constants.TOPIC_MODEL_TYPE_FIELD: topic_model_type,
            Constants.BOW_TYPE_FIELD: bow_type
        }

        print(new_dict)

        Constants.update_properties(new_dict)
        results = Constants.get_properties_copy()
        results.update(analyze_topics(include_stability=True))

        write_results_to_csv(csv_file_name, results)
        write_results_to_json(json_file_name, results)

        cycle_index += 1


def write_results_to_csv(file_name, results):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            w = csv.DictWriter(f, sorted(results.keys()))
            w.writeheader()
            w.writerow(results)
    else:
        with open(file_name, 'a') as f:
            w = csv.DictWriter(f, sorted(results.keys()))
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


def calculate_topic_stability_files(file_list):
    all_term_rankings = []

    for file_path in file_list:
        with open(file_path, 'rb') as read_file:
            topic_model = pickle.load(read_file)
            terms_matrix = get_topic_model_terms(
                topic_model, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
            all_term_rankings.append(terms_matrix)

    return calculate_stability(all_term_rankings)


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)


# fp1 = Constants.CACHE_FOLDER + 'restaurant_topic_model_nmf_separated_numtopics:28_iterations:200_passes:100_bow:NN_reviewtype:specific_document_level:1.pkl'
# fp2 = Constants.CACHE_FOLDER + 'restaurant_topic_model_nmf_full_numtopics:28_iterations:200_passes:100_bow:NN_reviewtype:specific_document_level:1.pkl'
#
# calculate_topic_stability_files([fp1, fp2])
