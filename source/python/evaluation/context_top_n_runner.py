import copy
import csv
import json
import os
import random
from subprocess import call
import time
import uuid
import gc
import numpy
import operator
# from fastFM import als
# from fastFM import mcmc
# from fastFM import sgd
from gensim import corpora

from etl import ETLUtils
from etl import libfm_converter
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
from evaluation import parameter_combinator
# from recommenders import fastfm_recommender
from topicmodeling.context import topic_model_creator
from tripadvisor.fourcity import extractor
from utils import utilities
from utils.constants import Constants

__author__ = 'fpena'


basic_headers = [
    Constants.RATING_FIELD,
    Constants.USER_ID_FIELD,
    Constants.ITEM_ID_FIELD
]


def build_headers(context_rich_topics):
    headers = basic_headers[:]
    for topic in context_rich_topics:
        topic_id = 'topic' + str(topic[0])
        headers.append(topic_id)
    if Constants.USE_NO_CONTEXT_TOPICS_SUM:
        headers.append('nocontexttopics')
    return headers


def run_libfm(train_file, test_file, predictions_file, log_file, save_file):

    libfm_command = Constants.LIBFM_FOLDER + 'libFM'

    command = [
        libfm_command,
        '-task',
        'r',
        '-method',
        Constants.FM_METHOD,
        '-regular',
        str(Constants.FM_REGULARIZATION0) + ',' +
        str(Constants.FM_REGULARIZATION1) + ',' +
        str(Constants.FM_REGULARIZATION2),
        '-learn_rate',
        str(Constants.FM_SDG_LEARN_RATE),
        '-train',
        train_file,
        '-test',
        test_file,
        '-dim',
        ','.join(map(str,
                     [Constants.FM_USE_BIAS, Constants.FM_USE_1WAY_INTERACTIONS,
                      Constants.FM_NUM_FACTORS])),
        '-init_stdev',
        str(Constants.FM_INIT_STDEV),
        '-iter',
        str(Constants.FM_ITERATIONS),
        '-out',
        predictions_file,
        '-save_model',
        save_file
    ]

    print(command)

    if Constants.LIBFM_SEED is not None:
        command.extend(['-seed', str(Constants.LIBFM_SEED)])

    f = open(log_file, "w")
    call(command, stdout=f)


def filter_reviews(records, reviews, review_type):
    print('filter: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    if not review_type:
        return records, reviews

    filtered_records = []
    filtered_reviews = []

    for record, review in zip(records, reviews):
        if record[Constants.PREDICTED_CLASS_FIELD] == review_type:
            filtered_records.append(record)
            filtered_reviews.append(review)

    return filtered_records, filtered_reviews


def write_results_to_csv(results):
    if not os.path.exists(Constants.CSV_RESULTS_FILE):
        with open(Constants.CSV_RESULTS_FILE, 'w') as f:
            w = csv.DictWriter(f, sorted(results.keys()))
            w.writeheader()
            w.writerow(results)
    else:
        with open(Constants.CSV_RESULTS_FILE, 'a') as f:
            w = csv.DictWriter(f, sorted(results.keys()))
            w.writerow(results)


def write_results_to_json(results):
    if not os.path.exists(Constants.JSON_RESULTS_FILE):
        with open(Constants.JSON_RESULTS_FILE, 'w') as f:
            json.dump(results, f)
            f.write('\n')
    else:
        with open(Constants.JSON_RESULTS_FILE, 'a') as f:
            json.dump(results, f)
            f.write('\n')


class ContextTopNRunner(object):

    def __init__(self):
        self.records = None
        self.original_records = None
        self.train_records = None
        self.test_records = None
        self.records_to_predict = None
        self.predictions = None
        self.top_n_evaluator = None
        self.headers = None
        self.important_records = None
        self.context_rich_topics = None
        self.context_topics_map = None
        self.csv_train_file = None
        self.csv_test_file = None
        self.context_predictions_file = None
        self.context_train_file = None
        self.context_test_file = None
        self.context_log_file = None
        self.libfm_model_file = None
        self.num_variables_in_model = None

    def clear(self):
        print('clear: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        # self.records = None
        self.train_records = None
        self.test_records = None
        self.records_to_predict = None
        self.top_n_evaluator = None
        self.headers = None
        self.important_records = None
        self.context_rich_topics = None
        self.context_topics_map = None

        if Constants.SOLVER == Constants.LIBFM:
            os.remove(self.csv_train_file)
            os.remove(self.csv_test_file)
            os.remove(self.context_predictions_file)
            os.remove(self.context_train_file)
            os.remove(self.context_test_file)
            os.remove(self.context_log_file)
            os.remove(self.libfm_model_file)

        self.csv_train_file = None
        self.csv_test_file = None
        self.context_predictions_file = None
        self.context_train_file = None
        self.context_test_file = None
        self.context_log_file = None
        self.libfm_model_file = None
        gc.collect()

    def create_tmp_file_names(self, cycle_index, fold_index):

        unique_id = uuid.uuid4().hex
        prefix = Constants.GENERATED_FOLDER + unique_id + '_' + \
            Constants.ITEM_TYPE

        print('unique id: %s' % unique_id)

        self.csv_train_file = prefix + '_train.csv'
        self.csv_test_file = prefix + '_test.csv'
        self.context_predictions_file = prefix + '_predictions.txt'
        self.context_train_file = self.csv_train_file + '.libfm'
        self.context_test_file = self.csv_test_file + '.libfm'
        self.context_log_file = prefix + '.log'
        self.libfm_model_file = prefix + '_trained_model.libfm'

        # self.csv_train_file = Constants.generate_file_name(
        #     'libfm_train', 'csv', Constants.GENERATED_FOLDER, cycle_index, fold_index, Constants.USE_CONTEXT)
        # self.csv_test_file = Constants.generate_file_name(
        #     'libfm_test', 'csv', Constants.GENERATED_FOLDER, cycle_index, fold_index, Constants.USE_CONTEXT)
        # self.context_predictions_file = Constants.generate_file_name(
        #     'libfm_predictions', 'txt', Constants.GENERATED_FOLDER, cycle_index, fold_index, Constants.USE_CONTEXT)
        # self.context_train_file = self.csv_train_file + '.libfm'
        # self.context_test_file = self.csv_test_file + '.libfm'
        # self.context_log_file = Constants.generate_file_name(
        #     'libfm_log', 'log', Constants.GENERATED_FOLDER, cycle_index, fold_index, Constants.USE_CONTEXT)
        # self.libfm_model_file = Constants.generate_file_name(
        #     'libfm_model', 'csv', Constants.GENERATED_FOLDER, cycle_index, fold_index, Constants.USE_CONTEXT)

    def load(self):
        print('load: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
            self.original_records =\
                ETLUtils.load_json_file(
                    Constants.RECSYS_CONTEXTUAL_PROCESSED_RECORDS_FILE)
        else:
            self.original_records =\
                ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)

        print('num_records: %d' % len(self.original_records))
        user_ids = extractor.get_groupby_list(
            self.original_records, Constants.USER_ID_FIELD)
        item_ids = extractor.get_groupby_list(
            self.original_records, Constants.ITEM_ID_FIELD)
        print('total users', len(user_ids))
        print('total items', len(item_ids))

    def shuffle(self, records):
        print('shuffle: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        random.shuffle(records)

    def get_records_to_predict_topn(self):
        print('get_records_to_predict_topn: %s'
              % time.strftime("%Y/%m/%d-%H:%M:%S"))

        self.top_n_evaluator = TopNEvaluator(
            self.records, self.test_records, Constants.ITEM_TYPE, 10,
            Constants.TOPN_NUM_ITEMS)
        self.top_n_evaluator.initialize()
        self.important_records = self.top_n_evaluator.important_records

        if Constants.TEST_CONTEXT_REVIEWS_ONLY:
            self.important_records = ETLUtils.filter_records(
                self.important_records, Constants.HAS_CONTEXT_FIELD, [True])

            self.records_to_predict =\
                self.top_n_evaluator.get_records_to_predict()

        if Constants.MAX_SAMPLE_TEST_SET is not None:
            print('important_records %d' % len(self.important_records))
            if len(self.important_records) > Constants.MAX_SAMPLE_TEST_SET:
                self.important_records = random.sample(
                    self.important_records, Constants.MAX_SAMPLE_TEST_SET)
            else:
                message = 'WARNING max_sample_test_set is greater than the ' \
                          'number of important records'
                print(message)

        self.top_n_evaluator.important_records = self.important_records
        self.records_to_predict = self.top_n_evaluator.get_records_to_predict()
        self.test_records = None
        gc.collect()

    def get_records_to_predict_rmse(self):
        print(
            'get_records_to_predict_rmse: %s' %
            time.strftime("%Y/%m/%d-%H:%M:%S")
        )
        self.important_records = self.test_records

        if Constants.TEST_CONTEXT_REVIEWS_ONLY:
            self.important_records = ETLUtils.filter_records(
                self.important_records, Constants.HAS_CONTEXT_FIELD, [True])

        self.records_to_predict = self.important_records
        self.test_records = None
        gc.collect()

    def get_records_to_predict(self, use_random_seeds):

        if use_random_seeds:
            utilities.plant_seeds()

        if Constants.EVALUATION_METRIC == 'topn_recall':
            self.get_records_to_predict_topn()
        elif Constants.EVALUATION_METRIC in ['rmse', 'mae']:
            self.get_records_to_predict_rmse()
        else:
            raise ValueError('Unrecognized evaluation metric')

    def train_topic_model(self, cycle_index, fold_index):

        context_extractor = topic_model_creator.create_topic_model(
            self.train_records, cycle_index, fold_index)
        self.context_rich_topics = context_extractor.context_rich_topics

        topics_file_path = Constants.generate_file_name(
            'context_topics', 'json', Constants.CACHE_FOLDER,
            cycle_index, fold_index, True)
        ETLUtils.save_json_file(
            topics_file_path, [dict(self.context_rich_topics)])
        print('Trained Context Extractor: %s' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        return context_extractor

    def load_context_reviews(self, cycle_index, fold_index):

        train_records_file_path = Constants.generate_file_name(
            'context_train_records', 'json', Constants.CACHE_FOLDER,
            cycle_index, fold_index, True)
        important_records_file_path = Constants.generate_file_name(
            'context_important_records', 'json', Constants.CACHE_FOLDER,
            cycle_index, fold_index, True)

        self.train_records = ETLUtils.load_json_file(train_records_file_path)
        self.important_records = \
            ETLUtils.load_json_file(important_records_file_path)
        self.load_cache_context_topics(cycle_index, fold_index)

        self.context_topics_map = {}
        for record in self.important_records:
            self.context_topics_map[record[Constants.REVIEW_ID_FIELD]] = \
                record[Constants.CONTEXT_TOPICS_FIELD]

        # self.train_records = self.filter_context_words(self.train_records)
        # self.print_context_topics(self.important_records)

        self.important_records = None
        gc.collect()

    def load_cache_context_topics(self, cycle_index, fold_index):

        print('load cache context topics: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        topics_file_path = Constants.generate_file_name(
            'context_topics', 'json', Constants.CACHE_FOLDER,
            cycle_index, fold_index, True)

        self.context_rich_topics = sorted(
            ETLUtils.load_json_file(topics_file_path)[0].items(),
            key=operator.itemgetter(1), reverse=True)

        self.context_topics_map = {}
        for record in self.important_records:
            self.context_topics_map[record[Constants.REVIEW_ID_FIELD]] = \
                record[Constants.CONTEXT_TOPICS_FIELD]

    def find_reviews_topics(self, context_extractor, cycle_index, fold_index):
        print('find topics: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        train_records_file_path = Constants.generate_file_name(
            'context_train_records', 'json', Constants.CACHE_FOLDER,
            cycle_index, fold_index, Constants.USE_CONTEXT)

        if os.path.exists(train_records_file_path):
            self.train_records = \
                ETLUtils.load_json_file(train_records_file_path)
        else:
            context_extractor.find_contextual_topics(self.train_records)
            ETLUtils.save_json_file(train_records_file_path, self.train_records)
        context_extractor.find_contextual_topics(
            self.important_records, Constants.TEXT_SAMPLING_PROPORTION)

        self.context_topics_map = {}
        for record in self.important_records:
            self.context_topics_map[record[Constants.REVIEW_ID_FIELD]] = \
                record[Constants.CONTEXT_TOPICS_FIELD]

        self.important_records = None
        gc.collect()

    @staticmethod
    def print_context_topics(records):
        dictionary = corpora.Dictionary.load(Constants.DICTIONARY_FILE)

        all_context_topics = set()

        for record in records:
            words = []
            corpus = record[Constants.CORPUS_FIELD]

            for element in corpus:
                word_id = element[0]
                word = dictionary[word_id]
                words.append(word + ' (' + str(word_id) + ')')

            context_topics = record[Constants.CONTEXT_TOPICS_FIELD]
            used_context_topics =\
                dict((k, v) for k, v in context_topics.items() if v > 0.0)
            all_context_topics |= set(used_context_topics.keys())

            print('words: %s' % ', '.join(words))
            print('text: %s' % record[Constants.TEXT_FIELD])
            print('bow', record[Constants.BOW_FIELD])
            # print('pos tags', record[Constants.POS_TAGS_FIELD])
            print(record[Constants.TOPICS_FIELD])
            # # print(record[Constants.CONTEXT_TOPICS_FIELD])
            print(used_context_topics)
            # print('')

        # print('important records: %d' % len(records))
        # print('context records: %d' % len(context_records))
        # print('no context records: %d' % len(no_context_records))
        # print('all used context words', all_context_words)
        print('all used context topics', all_context_topics)
        # print('all used context words count: %d' % len(all_context_words))
        print('all used context topics: %d' % len(all_context_topics))

    def prepare_records_for_libfm(self):
        print('prepare_records_for_libfm: %s' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        self.headers = build_headers(self.context_rich_topics)

        if Constants.FM_REVIEW_TYPE == Constants.SPECIFIC or \
                Constants.FM_REVIEW_TYPE == Constants.GENERIC:
            self.train_records = ETLUtils.filter_records(
                self.train_records, Constants.PREDICTED_CLASS_FIELD,
                [Constants.FM_REVIEW_TYPE])

        with open(self.csv_train_file, 'w') as out_file:
            writer = csv.writer(out_file)

            # Write header
            writer.writerow(self.headers)

            for record in self.train_records:
                row = []
                for header in basic_headers:
                    row.append(record[header])

                if Constants.USE_CONTEXT is True:
                    context_topics = record[Constants.CONTEXT_TOPICS_FIELD]
                    for topic in self.context_rich_topics:
                        # print('context_topics', context_topics)
                        row.append(context_topics['topic' + str(topic[0])])

                    if Constants.USE_NO_CONTEXT_TOPICS_SUM:
                        row.append(context_topics['nocontexttopics'])

                writer.writerow(row)

        self.train_records = None
        gc.collect()

        with open(self.csv_test_file, 'w') as out_file:
            writer = csv.writer(out_file)

            # Write header
            writer.writerow(self.headers)

            for record in self.records_to_predict:
                row = []
                for header in basic_headers:
                    row.append(record[header])

                if Constants.USE_CONTEXT is True:
                    important_record = record[Constants.REVIEW_ID_FIELD]
                    context_topics = \
                        self.context_topics_map[important_record]
                    for topic in self.context_rich_topics:
                        row.append(context_topics['topic' + str(topic[0])])

                    if Constants.USE_NO_CONTEXT_TOPICS_SUM:
                        row.append(context_topics['nocontexttopics'])

                writer.writerow(row)

        # self.records_to_predict = None
        self.context_topics_map = None
        self.context_rich_topics = None
        gc.collect()

        print('Exported CSV and JSON files: %s'
              % time.strftime("%Y/%m/%d-%H:%M:%S"))

        csv_files = [
            self.csv_train_file,
            self.csv_test_file
        ]

        print('num_cols', len(self.headers))

        self.num_variables_in_model = libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], [], ',', has_header=True,
            suffix='.libfm')

        print('Exported LibFM files: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    # def predict_fastfm(self):
    #
    #     if Constants.USE_CONTEXT:
    #         for record in self.records_to_predict:
    #             important_record = record[Constants.REVIEW_ID_FIELD]
    #             record[Constants.CONTEXT_TOPICS_FIELD] = \
    #                 self.context_topics_map[important_record]
    #
    #     all_records = self.train_records + self.records_to_predict
    #     x_matrix, y_vector = fastfm_recommender.records_to_matrix(
    #         all_records, self.context_rich_topics)
    #
    #     encoder = OneHotEncoder(categorical_features=[0, 1], sparse=True)
    #     encoder.fit(x_matrix)
    #
    #     x_train = encoder.transform(x_matrix[:len(self.train_records)])
    #     y_train = y_vector[:len(self.train_records)]
    #     x_test = encoder.transform(x_matrix[len(self.train_records):])
    #
    #     if Constants.FASTFM_METHOD == 'mcmc':
    #         # solver = mcmc.FMRegression(n_iter=num_iters, rank=num_factors)
    #         solver = mcmc.FMRegression(rank=Constants.FM_NUM_FACTORS)
    #         self.predictions = solver.fit_predict(x_train, y_train, x_test)
    #     elif Constants.FASTFM_METHOD == 'als':
    #         solver = als.FMRegression(rank=Constants.FM_NUM_FACTORS)
    #         solver.fit(x_train, y_train)
    #         self.predictions = solver.predict(x_test)
    #     elif Constants.FASTFM_METHOD == 'sgd':
    #         solver = sgd.FMRegression(rank=Constants.FM_NUM_FACTORS)
    #         solver.fit(x_train, y_train)
    #         self.predictions = solver.predict(x_test)

    def predict_libfm(self):
        print('predict: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        run_libfm(
            self.context_train_file, self.context_test_file,
            self.context_predictions_file, self.context_log_file,
            self.libfm_model_file
        )
        self.predictions = rmse_calculator.read_targets_from_txt(
            self.context_predictions_file)

    def predict(self):
        if Constants.SOLVER == Constants.LIBFM:
            self.prepare_records_for_libfm()
            self.predict_libfm()
        # elif Constants.SOLVER == Constants.FASTFM:
        #     self.predict_fastfm()

    def evaluate_topn(self):
        print('evaluate_topn: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        self.top_n_evaluator.evaluate(self.predictions)
        recall = self.top_n_evaluator.recall

        print('Recall: %f' % recall)
        print('Specific recall: %f' % self.top_n_evaluator.specific_recall)
        print('Generic recall: %f' % self.top_n_evaluator.generic_recall)

        results = {
            Constants.TOPN_RECALL: self.top_n_evaluator.recall,
            Constants.SPECIFIC + '_' + Constants.TOPN_RECALL:
                self.top_n_evaluator.specific_recall,
            Constants.GENERIC + '_' + Constants.TOPN_RECALL:
                self.top_n_evaluator.generic_recall,
            Constants.HAS_CONTEXT + '_' + Constants.TOPN_RECALL:
                self.top_n_evaluator.has_context_recall,
            Constants.HAS_NO_CONTEXT + '_' + Constants.TOPN_RECALL:
                self.top_n_evaluator.has_no_context_recall,
        }

        return results

    def evaluate_rmse(self):
        print('evaluate_rmse: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        true_values = [
            record[Constants.RATING_FIELD] for record in self.records_to_predict
        ]

        specific_true_values = []
        specific_predictions = []
        generic_true_values = []
        generic_predictions = []
        has_context_true_values = []
        has_context_predictions = []
        has_no_context_true_values = []
        has_no_context_predictions = []

        index = 0
        for record, prediction in zip(
                self.records_to_predict, self.predictions):
            if record[Constants.PREDICTED_CLASS_FIELD] == 'specific':
                specific_true_values.append(record[Constants.RATING_FIELD])
                specific_predictions.append(prediction)
            elif record[Constants.PREDICTED_CLASS_FIELD] == 'generic':
                generic_true_values.append(record[Constants.RATING_FIELD])
                generic_predictions.append(prediction)
            if record[Constants.HAS_CONTEXT_FIELD]:
                has_context_true_values.append(record[Constants.RATING_FIELD])
                has_context_predictions.append(prediction)
            else:
                has_no_context_true_values.append(record[Constants.RATING_FIELD])
                has_no_context_predictions.append(prediction)
            index += 1

        metric = Constants.EVALUATION_METRIC

        overall_result = None
        specific_result = None
        generic_result = None
        has_context_result = None
        has_no_context_result = None

        if metric == 'rmse':
            overall_result = \
                rmse_calculator.calculate_rmse(true_values, self.predictions)
            specific_result = rmse_calculator.calculate_rmse(
                specific_true_values, specific_predictions)
            generic_result = rmse_calculator.calculate_rmse(
                generic_true_values, generic_predictions)
            has_context_result = rmse_calculator.calculate_rmse(
                has_context_true_values, has_context_predictions)
            has_no_context_result = rmse_calculator.calculate_rmse(
                has_no_context_true_values, has_no_context_predictions)
        elif metric == 'mae':
            overall_result = \
                rmse_calculator.calculate_mae(true_values, self.predictions)
            specific_result = rmse_calculator.calculate_mae(
                specific_true_values, specific_predictions)
            generic_result = rmse_calculator.calculate_mae(
                generic_true_values, generic_predictions)
            has_context_result = rmse_calculator.calculate_mae(
                has_context_true_values, has_context_predictions)
            has_no_context_result = rmse_calculator.calculate_mae(
                has_no_context_true_values, has_no_context_predictions)

        print(metric + ': %f' % overall_result)
        print('Specific ' + metric + ': %f' % specific_result)
        print('Generic ' + metric + ': %f' % generic_result)
        print('Has context ' + metric + ': %f' % has_context_result)
        print('Has no ' + metric + ': %f' % has_no_context_result)

        results = {
            metric: overall_result,
            Constants.SPECIFIC + '_' + metric: specific_result,
            Constants.GENERIC + '_' + metric: generic_result,
            Constants.HAS_CONTEXT + '_' + metric: has_context_result,
            Constants.HAS_NO_CONTEXT + '_' + metric: has_no_context_result,
        }

        return results

    def evaluate(self):

        if Constants.EVALUATION_METRIC == 'topn_recall':
            return self.evaluate_topn()
        elif Constants.EVALUATION_METRIC in ['rmse', 'mae']:
            return self.evaluate_rmse()
        else:
            raise ValueError('Unrecognized evaluation metric')

    def perform_cross_validation(self, records):

        Constants.print_properties()

        # self.plant_seeds()

        metrics_list = []
        total_cycle_time = 0.0
        num_cycles = Constants.NUM_CYCLES
        num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
        total_iterations = num_cycles * num_folds
        split = 1 - (1/float(num_folds))
        metric_name = Constants.EVALUATION_METRIC

        # self.load()

        for i in range(num_cycles):

            print('\n\nCycle: %d/%d' % ((i+1), num_cycles))

            self.records = copy.deepcopy(records)
            if Constants.SHUFFLE_DATA:
                self.shuffle(self.records)

            for j in range(num_folds):

                fold_start = time.time()
                cv_start = float(j) / num_folds
                print('\nFold: %d/%d' % ((j+1), num_folds))

                self.create_tmp_file_names(i, j)
                self.train_records, self.test_records = \
                    ETLUtils.split_train_test_copy(
                        self.records, split=split, start=cv_start)
                # subsample_size = int(len(self.train_records)*0.5)
                # self.train_records = self.train_records[:subsample_size]
                self.get_records_to_predict(True)
                if Constants.USE_CONTEXT:
                    if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
                        self.load_cache_context_topics(None, None)
                    else:
                        if Constants.CACHE_CONTEXT_REVIEWS:
                            self.load_context_reviews(i, j)
                        else:
                            context_extractor = self.train_topic_model(i, j)
                            self.find_reviews_topics(context_extractor, i, j)
                else:
                    self.context_rich_topics = []
                self.predict()
                metrics = self.evaluate()

                metrics_list.append(metrics)
                print('Accumulated %s: %f' % (metric_name,
                    numpy.mean([k[metric_name] for k in metrics_list])))

                fold_end = time.time()
                fold_time = fold_end - fold_start
                total_cycle_time += fold_time
                self.clear()
                print("Total fold %d time = %f seconds" % ((j+1), fold_time))

        results = self.summarize_results(metrics_list)

        average_cycle_time = total_cycle_time / total_iterations
        results['cycle_time'] = average_cycle_time
        print('average cycle time: %f' % average_cycle_time)

        write_results_to_csv(results)
        write_results_to_json(results)

        return results

    @staticmethod
    def summarize_results(metrics_list):

        metric_name = Constants.EVALUATION_METRIC
        specific_metric_name = Constants.SPECIFIC + '_' + metric_name
        generic_metric_name = Constants.GENERIC + '_' + metric_name
        has_context_metric_name = Constants.HAS_CONTEXT + '_' + metric_name
        has_no_context_metric_name = Constants.HAS_NO_CONTEXT + '_' + metric_name

        metric_average = \
            numpy.mean(numpy.mean([k[metric_name] for k in metrics_list]))
        metric_stdev = numpy.std([k[metric_name] for k in metrics_list])
        average_specific_metric = numpy.mean(
            [k[specific_metric_name] for k in metrics_list])
        average_generic_metric = numpy.mean(
            [k[generic_metric_name] for k in metrics_list])
        average_has_context_metric = numpy.mean(
            [k[has_context_metric_name] for k in metrics_list])
        average_has_no_context_metric = numpy.mean(
            [k[has_no_context_metric_name] for k in metrics_list])

        print('average %s:\t\t\t%f' % (metric_name, metric_average))
        print('average specific %s:\t%f' % (
            metric_name, average_specific_metric))
        print('average generic %s:\t%f' % (
            metric_name, average_generic_metric))
        print('average has context %s:\t%f' % (
            metric_name, average_has_context_metric))
        print('average has no context %s:\t%f' % (
            metric_name, average_has_no_context_metric))
        print('standard deviation %s:\t%f (%f%%)' % (
            metric_name, metric_stdev, (metric_stdev / metric_average * 100)))
        print('End: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        #
        results = Constants.get_properties_copy()
        results[metric_name] = metric_average
        results[specific_metric_name] = average_specific_metric
        results[generic_metric_name] = average_generic_metric
        results[has_context_metric_name] = average_has_context_metric
        results[has_no_context_metric_name] = average_has_no_context_metric
        results[metric_name + '_stdev'] = metric_stdev
        results['timestamp'] = time.strftime("%Y/%m/%d-%H:%M:%S")

        write_results_to_csv(results)
        write_results_to_json(results)

        return results

    def run_single_fold(self, parameters):

        fold = parameters['fold']

        Constants.update_properties(parameters)

        Constants.print_properties()

        utilities.plant_seeds()
        self.load()

        records = self.original_records

        # self.plant_seeds()
        total_cycle_time = 0.0
        num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
        split = 1 - (1 / float(num_folds))
        self.records = copy.deepcopy(records)
        if Constants.SHUFFLE_DATA:
            self.shuffle(self.records)

        fold_start = time.time()
        cv_start = float(fold) / num_folds
        print('\nFold: %d/%d' % ((fold + 1), num_folds))

        self.create_tmp_file_names(0, fold)
        self.train_records, self.test_records = \
            ETLUtils.split_train_test_copy(
                self.records, split=split, start=cv_start)
        # subsample_size = int(len(self.train_records)*0.5)
        # self.train_records = self.train_records[:subsample_size]
        self.get_records_to_predict(True)
        if Constants.USE_CONTEXT:
            if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
                self.load_cache_context_topics(None, None)
            else:
                if Constants.CACHE_CONTEXT_REVIEWS:
                    self.load_context_reviews(0, fold)
                else:
                    context_extractor = self.train_topic_model(0, fold)
                    self.find_reviews_topics(context_extractor, 0, fold)
        else:
            self.context_rich_topics = []
        self.predict()
        metrics = self.evaluate()

        fold_end = time.time()
        fold_time = fold_end - fold_start
        total_cycle_time += fold_time
        self.clear()
        print("Total fold %d time = %f seconds" % ((fold + 1), fold_time))

        return metrics

    def run(self):

        utilities.plant_seeds()
        self.load()

        records = self.original_records

        if Constants.CROSS_VALIDATION_STRATEGY == 'nested_validate':
            num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
            cycle = Constants.NESTED_CROSS_VALIDATION_CYCLE
            split = 1 - (1/float(num_folds))
            cv_start = float(cycle) / num_folds
            print('cv_start', cv_start)
            records, _ = ETLUtils.split_train_test(
                self.original_records, split, cv_start)
            return self.perform_cross_validation(records)
        elif Constants.CROSS_VALIDATION_STRATEGY == 'nested_test':
            return self.perform_cross_validation(records)
        else:
            raise ValueError('Unknown cross-validation strategy')


def run_tests():

    combined_parameters = parameter_combinator.get_combined_parameters()

    test_cycle = 1
    num_tests = len(combined_parameters)
    highest_value = -1
    best_parameters = None
    for properties in combined_parameters:
        Constants.update_properties(properties)
        context_top_n_runner = ContextTopNRunner()

        print('\n\n******************\nTest %d/%d\n******************\n' %
              (test_cycle, num_tests))

        results = context_top_n_runner.run()
        if results[Constants.EVALUATION_METRIC] > highest_value:
            highest_value = results[Constants.EVALUATION_METRIC]
            best_parameters = properties
        test_cycle += 1

    print('highest %s: %f' % (Constants.EVALUATION_METRIC, highest_value))
    print(best_parameters)


def run_test_folds():
    context_parameters = [
        {'fold': 0, Constants.FM_NUM_FACTORS_FIELD: 16,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 1, Constants.FM_NUM_FACTORS_FIELD: 32,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 2, Constants.FM_NUM_FACTORS_FIELD: 1,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 3, Constants.FM_NUM_FACTORS_FIELD: 2,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 4, Constants.FM_NUM_FACTORS_FIELD: 16,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 5, Constants.FM_NUM_FACTORS_FIELD: 16,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 6, Constants.FM_NUM_FACTORS_FIELD: 32,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 7, Constants.FM_NUM_FACTORS_FIELD: 32,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 8, Constants.FM_NUM_FACTORS_FIELD: 32,
         Constants.USE_CONTEXT_FIELD: True},
        {'fold': 9, Constants.FM_NUM_FACTORS_FIELD: 32,
         Constants.USE_CONTEXT_FIELD: True}
    ]

    nocontext_parameters = [
        {'fold': 0, Constants.FM_NUM_FACTORS_FIELD: 128,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 1, Constants.FM_NUM_FACTORS_FIELD: 2,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 2, Constants.FM_NUM_FACTORS_FIELD: 128,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 3, Constants.FM_NUM_FACTORS_FIELD: 8,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 4, Constants.FM_NUM_FACTORS_FIELD: 32,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 5, Constants.FM_NUM_FACTORS_FIELD: 32,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 6, Constants.FM_NUM_FACTORS_FIELD: 2,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 7, Constants.FM_NUM_FACTORS_FIELD: 4,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 8, Constants.FM_NUM_FACTORS_FIELD: 4,
         Constants.USE_CONTEXT_FIELD: False},
        {'fold': 9, Constants.FM_NUM_FACTORS_FIELD: 1,
         Constants.USE_CONTEXT_FIELD: False}
    ]

    no_context_results = []
    context_results = []
    my_context_top_n_runner = ContextTopNRunner()
    for parameters in nocontext_parameters:
        result = my_context_top_n_runner.run_single_fold(parameters)
        no_context_results.append(result)
    for parameters in context_parameters:
        result = my_context_top_n_runner.run_single_fold(parameters)
        context_results.append(result)

    print('\nContext results')
    context_results = ContextTopNRunner.summarize_results(context_results)

    print('\nNo Context results')
    no_context_results = ContextTopNRunner.summarize_results(no_context_results)

    print('\nImprovement')
    metric_name = Constants.EVALUATION_METRIC
    specific_metric_name = Constants.SPECIFIC + '_' + metric_name
    generic_metric_name = Constants.GENERIC + '_' + metric_name
    has_context_metric_name = Constants.HAS_CONTEXT + '_' + metric_name
    has_no_context_metric_name = Constants.HAS_NO_CONTEXT + '_' + metric_name

    metric_improvement = \
        (context_results[metric_name] / no_context_results[metric_name] - 1) * 100
    specific_metric_improvement = \
        (context_results[specific_metric_name] / no_context_results[specific_metric_name] - 1) * 100
    generic_metric_improvement = \
        (context_results[generic_metric_name] / no_context_results[generic_metric_name] - 1) * 100
    has_context_metric_improvement = \
        (context_results[has_context_metric_name] / no_context_results[has_context_metric_name] - 1) * 100
    has_no_context_metric_improvement = \
        (context_results[has_no_context_metric_name] / no_context_results[has_no_context_metric_name] - 1) * 100
    print('%s improvement:\t\t\t%f%%' % (metric_name, metric_improvement))
    print('specific %s improvement:\t%f%%' % (metric_name, specific_metric_improvement))
    print('generic %s improvement:\t%f%%' % (metric_name, generic_metric_improvement))
    print('has context %s improvement:\t%f%%' % (metric_name, has_context_metric_improvement))
    print('has no context %s improvement:\t%f%%' % (metric_name, has_no_context_metric_improvement))
#

# start = time.time()
# my_context_top_n_runner = ContextTopNRunner()
# my_context_top_n_runner.run()
# run_tests()
# run_test_folds()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
