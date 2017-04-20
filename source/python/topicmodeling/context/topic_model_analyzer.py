import argparse
import csv
import json
import os
import random
import time

import itertools

import numpy
import xlsxwriter
from pandas import DataFrame

from etl import ETLUtils
from topicmodeling.context import topic_model_creator
from topicmodeling.context.context_extractor import ContextExtractor
from topicmodeling.hungarian import HungarianError
from topicmodeling.jaccard_similarity import AverageJaccard
from topicmodeling.jaccard_similarity import RankingSetAgreement
from topicmodeling.nmf_topic_extractor import NmfTopicExtractor
from utils import utilities
from utils.constants import Constants
from utils.utilities import grouped_hotel_context_words, \
    grouped_restaurant_context_words


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
    rank_score = 0.0
    num_terms = Constants.TOPIC_MODEL_STABILITY_NUM_TERMS
    for topic_word in topic_words:
        word = topic_word.split('*')[1]
        word_probability_score = float(topic_word.split('*')[0])
        words_dict['word' + str(index)] = topic_word.encode('utf-8')
        if word in my_context_words:
            probability_score += word_probability_score
            word_rank_score = num_terms - index if index < num_terms else 0
            rank_score += word_rank_score
        index += 1

    max_rank_score = float(num_terms * (num_terms + 1) / 2)
    rank_score /= max_rank_score

    words_dict['probability_score'] = probability_score
    words_dict['rank_score'] = rank_score

    return words_dict


def analyze_topics(include_stability=True):

    start_time = time.time()

    utilities.plant_seeds()
    records = \
        ETLUtils.load_json_file(Constants.RECSYS_TOPICS_PROCESSED_RECORDS_FILE)
    print('num_reviews', len(records))
    num_topics = Constants.TOPIC_MODEL_NUM_TOPICS
    num_terms = Constants.TOPIC_MODEL_STABILITY_NUM_TERMS

    if Constants.TOPIC_MODEL_TYPE == 'ensemble':
        topic_model = NmfTopicExtractor()
        topic_model.load_trained_data()
        topic_model_string = topic_model.print_topic_model(num_terms)
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
    sample_ratio = Constants.TOPIC_MODEL_STABILITY_SAMPLE_RATIO
    if include_stability:
        stability = calculate_topic_stability(records, sample_ratio).mean()
    scores['stability'] = stability

    # separation_score =\
    #     (high_ratio_mean_score / low_ratio_mean_score)\
    #     if low_ratio_mean_score != 0\
    #     else 'N/A'
    # rank_separation_score =\
    #     (high_rank_ratio_mean_score / low_rank_ratio_mean_score)\
    #     if low_rank_ratio_mean_score != 0\
    #     else 'N/A'
    gamma = 0.5
    separation_score = gamma*high_ratio_mean_score + (1 - gamma)*(1-low_ratio_mean_score)
    rank_separation_score = gamma*high_rank_ratio_mean_score + (1 - gamma)*(1-low_rank_ratio_mean_score)
    joint_separation_score =\
        (high_ratio_mean_score + (1 - low_ratio_mean_score)) / 2
    joint_rank_separation_score =\
        (high_rank_ratio_mean_score +
         (1 - low_rank_ratio_mean_score)) / 2
    scores['separation_score'] = separation_score
    scores['rank_separation_score'] = rank_separation_score
    scores['joint_separation_score'] = joint_separation_score
    scores['joint_rank_separation_score'] = joint_rank_separation_score
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


def calculate_topic_stability(records, sample_ratio=None):

    Constants.update_properties({
        Constants.NUMPY_RANDOM_SEED_FIELD: Constants.NUMPY_RANDOM_SEED + 10,
        Constants.RANDOM_SEED_FIELD: Constants.RANDOM_SEED + 10
    })
    utilities.plant_seeds()
    Constants.print_properties()

    if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
        num_records = len(records)
        records = records[:num_records / 2]
    print('num_reviews', len(records))

    all_term_rankings = []

    context_extractor =\
        topic_model_creator.create_topic_model(records, None, None)
    terms_matrix = get_topic_model_terms(
        context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
    all_term_rankings.append(terms_matrix)

    for _ in range(Constants.TOPIC_MODEL_STABILITY_ITERATIONS - 1):

        if sample_ratio is None:
            sampled_records = records
        else:
            sampled_records = sample_list(records, sample_ratio)
        context_extractor = \
            topic_model_creator.train_context_extractor(sampled_records, False)
        terms_matrix = get_topic_model_terms(
            context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
        all_term_rankings.append(terms_matrix)

    return calculate_stability(all_term_rankings)


def sample_list(lst, sample_ratio):

    num_samples = int(len(lst) * sample_ratio)
    sampled_list = [
        lst[i] for i in sorted(random.sample(xrange(len(lst)), num_samples))]

    return sampled_list


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


def generate_excel_file(records, file_name=None):
    my_context_words = []
    if 'hotel' in Constants.ITEM_TYPE:
        for values in grouped_hotel_context_words.values():
            my_context_words.extend(values)
    elif 'restaurant' in Constants.ITEM_TYPE:
        for values in grouped_restaurant_context_words.values():
            my_context_words.extend(values)

    if file_name is None:
        file_name = Constants.generate_file_name(
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

    csv_file_name = Constants.generate_file_name(
        'topic_model_analysis', 'csv', Constants.DATASET_FOLDER, None, None,
        False)
    json_file_name = Constants.generate_file_name(
        'topic_model_analysis', 'json', Constants.DATASET_FOLDER, None, None,
        False)
    print(csv_file_name)

    # export_lda_topics(0, 0)
    # epsilon_list = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.35, 0.5]
    epsilon_list = [0.05]
    alpha_list = [0.0]
    # num_topics_list =\
    #     [5, 10, 35, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    # num_topics_list = [10, 20, 30, 50, 75, 100, 150, 300]
    # num_topics_list = [150, 300]
    num_topics_list = range(1, 51)
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


def manual_main():

    csv_file_name = Constants.generate_file_name(
        'topic_model_analysis', 'csv', Constants.DATASET_FOLDER, None, None,
        False)
    json_file_name = Constants.generate_file_name(
        'topic_model_analysis', 'json', Constants.DATASET_FOLDER, None, None,
        False)
    print(json_file_name)
    print(csv_file_name)

    num_topics_list = [Constants.TOPIC_MODEL_NUM_TOPICS]
    num_cycles = len(num_topics_list)
    cycle_index = 1
    for num_topics in num_topics_list:
        print('\ncycle_index: %d/%d' % (cycle_index, num_cycles))
        new_dict = {
            Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: num_topics,
        }

        print(new_dict)

        Constants.update_properties(new_dict)
        results = Constants.get_properties_copy()
        results.update(analyze_topics(include_stability=False))

        write_results_to_csv(csv_file_name, results)
        write_results_to_json(json_file_name, results)

        cycle_index += 1


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--numtopics', metavar='int', type=int,
        nargs=1, help='The number of topics of the topic model')

    args = parser.parse_args()
    num_topics = args.numtopics[0] if args.numtopics is not None else None

    if num_topics is not None:
        Constants.update_properties(
            {Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: num_topics})

    results = Constants.get_properties_copy()
    results.update(analyze_topics(include_stability=True))

    csv_file_name = Constants.generate_file_name(
        'topic_model_analysis', 'csv', Constants.DATASET_FOLDER, None, None,
        False)
    json_file_name = Constants.generate_file_name(
        'topic_model_analysis', 'json', Constants.DATASET_FOLDER, None, None,
        False)

    write_results_to_csv(csv_file_name, results)
    write_results_to_json(json_file_name, results)


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

start = time.time()
manual_main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)


# if __name__ == '__main__':
#     start = time.time()
#     cli_main()
#     end = time.time()
#     total_time = end - start
#     print("Total time = %f seconds" % total_time)
