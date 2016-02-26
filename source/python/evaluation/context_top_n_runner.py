import copy
import csv
from multiprocessing import Pool
import os
import random
from subprocess import call
import time
import cPickle as pickle
import traceback
import uuid
import itertools
import numpy
from etl import ETLUtils
from etl import libfm_converter
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
from topicmodeling.context.lda_based_context import LdaBasedContext
from tripadvisor.fourcity import extractor
from utils.constants import Constants

__author__ = 'fpena'


def build_headers(context_rich_topics):
    headers = [
        Constants.RATING_FIELD,
        Constants.USER_ID_FIELD,
        Constants.ITEM_ID_FIELD
    ]
    for topic in context_rich_topics:
        topic_id = 'topic' + str(topic[0])
        headers.append(topic_id)
    return headers


def create_user_item_map(records):
    user_ids = extractor.get_groupby_list(records, Constants.USER_ID_FIELD)
    user_item_map = {}
    user_count = 0

    for user_id in user_ids:
        user_records =\
            ETLUtils.filter_records(records, Constants.USER_ID_FIELD, [user_id])
        user_items =\
            extractor.get_groupby_list(user_records, Constants.ITEM_ID_FIELD)
        user_item_map[user_id] = user_items
        user_count += 1

        # print("user count %d" % user_count),
        print 'user count: {0}\r'.format(user_count),

    print

    return user_item_map


def run_libfm(train_file, test_file, predictions_file, log_file):

    libfm_command = Constants.LIBFM_FOLDER + 'libFM'

    command = [
        libfm_command,
        '-task',
        'r',
        '-train',
        train_file,
        '-test',
        test_file,
        '-dim',
        '1,1,8',
        '-out',
        predictions_file
    ]

    if Constants.LIBFM_SEED is not None:
        command.extend(['-seed', str(Constants.LIBFM_SEED)])

    f = open(log_file, "w")
    call(command, stdout=f)


def filter_reviews(records, reviews, review_type):
    print('filter: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    if not review_type:
        return records, reviews

    filtered_records = []
    filtered_reviews = []

    for record, review in zip(records, reviews):
        if record[Constants.PREDICTED_CLASS_FIELD] == review_type:
            filtered_records.append(record)
            filtered_reviews.append(review)

    return filtered_records, filtered_reviews


class ContextTopNRunner(object):

    def __init__(self):
        self.records = None
        self.original_records = None
        self.train_records = None
        self.test_records = None
        self.records_to_predict = None
        self.top_n_evaluator = None
        self.headers = None
        self.important_records = None
        self.context_rich_topics = []
        self.csv_train_file = None
        self.csv_test_file = None
        self.context_predictions_file = None
        self.context_train_file = None
        self.context_test_file = None
        self.context_log_file = None

    def clear(self):
        print('clear: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        # self.records = None
        self.train_records = None
        self.test_records = None
        self.records_to_predict = None
        self.top_n_evaluator = None
        self.headers = None
        self.important_records = None
        self.context_rich_topics = None

        os.remove(self.csv_train_file)
        os.remove(self.csv_test_file)
        os.remove(self.context_predictions_file)
        os.remove(self.context_train_file)
        os.remove(self.context_test_file)
        os.remove(self.context_log_file)

        self.csv_train_file = None
        self.csv_test_file = None
        self.context_predictions_file = None
        self.context_train_file = None
        self.context_test_file = None
        self.context_log_file = None

    def create_tmp_file_names(self):

        unique_id = uuid.uuid4().hex
        prefix = Constants.GENERATED_FOLDER + unique_id + '_' + \
                 Constants.ITEM_TYPE
        # prefix = constants.GENERATED_FOLDER + constants.ITEM_TYPE

        # print('unique id: %s' % unique_id)

        self.csv_train_file = prefix + '_train.csv'
        self.csv_test_file = prefix + '_test.csv'
        self.context_predictions_file = prefix + '_predictions.txt'
        self.context_train_file = self.csv_train_file + '.libfm'
        self.context_test_file = self.csv_test_file + '.libfm'
        self.context_log_file = prefix + '.log'

    @staticmethod
    def plant_seeds():

        if Constants.RANDOM_SEED is not None:
            print('random seed: %d' % Constants.RANDOM_SEED)
            random.seed(Constants.RANDOM_SEED)
        if Constants.NUMPY_RANDOM_SEED is not None:
            print('numpy random seed: %d' % Constants.NUMPY_RANDOM_SEED)
            numpy.random.seed(Constants.NUMPY_RANDOM_SEED)

    def load(self):
        print('load: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        self.original_records = ETLUtils.load_json_file(Constants.RECORDS_FILE)
        print('num_records: %d' % len(self.original_records))

        if not os.path.exists(Constants.USER_ITEM_MAP_FILE):
            records = ETLUtils.load_json_file(Constants.RECORDS_FILE)
            user_item_map = create_user_item_map(records)
            with open(Constants.USER_ITEM_MAP_FILE, 'wb') as write_file:
                pickle.dump(user_item_map, write_file, pickle.HIGHEST_PROTOCOL)

    def shuffle(self):
        print('shuffle: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        random.shuffle(self.records)

    def split(self):
        print('split: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        num_records = len(self.records)
        num_split_records =\
            int(float(Constants.SPLIT_PERCENTAGE) / 100 * num_records)
        self.train_records = self.records[:num_split_records]
        self.test_records = self.records[num_split_records:]

    def export(self):
        print('export: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        if Constants.REVIEW_TYPE:
            self.records = ETLUtils.filter_records(
                self.records, Constants.PREDICTED_CLASS_FIELD,
                [Constants.REVIEW_TYPE])
            self.test_records = ETLUtils.filter_records(
                self.test_records, Constants.PREDICTED_CLASS_FIELD,
                [Constants.REVIEW_TYPE])

        with open(Constants.USER_ITEM_MAP_FILE, 'rb') as read_file:
            user_item_map = pickle.load(read_file)

        self.top_n_evaluator = TopNEvaluator(
            self.records, self.test_records, Constants.ITEM_TYPE, 10,
            Constants.TOPN_NUM_ITEMS)
        self.top_n_evaluator.initialize(user_item_map)
        self.records_to_predict = self.top_n_evaluator.get_records_to_predict()
        # self.top_n_evaluator.export_records_to_predict(RECORDS_TO_PREDICT_FILE)
        self.important_records = self.top_n_evaluator.important_records

    def train_topic_model(self):
        print('train topic model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        # self.train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
        lda_based_context = LdaBasedContext(self.train_records)
        lda_based_context.get_context_rich_topics()
        self.context_rich_topics = lda_based_context.context_rich_topics
        print('Trained LDA Model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        self.context_rich_topics = lda_based_context.context_rich_topics

        return lda_based_context

    def find_reviews_topics(self, lda_based_context):
        print('find topics: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        lda_based_context.find_contextual_topics(self.train_records)
        # lda_based_context.find_contextual_topics(self.records_to_predict)

        topics_map = {}
        lda_based_context.find_contextual_topics(self.important_records)
        for record in self.important_records:
            topics_map[record[Constants.REVIEW_ID_FIELD]] =\
                record[Constants.TOPICS_FIELD]

        for record in self.records_to_predict:
            topic_distribution = topics_map[record[Constants.REVIEW_ID_FIELD]]

            context_topics = {}
            for i in self.context_rich_topics:
                topic_id = 'topic' + str(i[0])
                context_topics[topic_id] = topic_distribution[i[0]]

            record[Constants.CONTEXT_TOPICS_FIELD] = context_topics

        print('contextual test set size: %d' % len(self.records_to_predict))
        print('Exported contextual topics: %s' %
              time.strftime("%Y/%d/%m-%H:%M:%S"))

        return self.train_records, self.records_to_predict

    def prepare(self):
        print('prepare: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        self.headers = build_headers(self.context_rich_topics)

        contextual_train_set = copy.deepcopy(self.train_records)
        contextual_test_set = copy.deepcopy(self.records_to_predict)

        if Constants.USE_CONTEXT is True:
            for record in contextual_train_set:
                record.update(record[Constants.CONTEXT_TOPICS_FIELD])

            for record in contextual_test_set:
                record.update(record[Constants.CONTEXT_TOPICS_FIELD])

            ETLUtils.drop_fields([Constants.TOPICS_FIELD], self.train_records)
            # ETLUtils.drop_fields([constants.TOPICS_FIELD], self.records_to_predict)

        contextual_train_set = ETLUtils.select_fields(self.headers, contextual_train_set)
        contextual_test_set = ETLUtils.select_fields(self.headers, contextual_test_set)

        ETLUtils.save_csv_file(
            self.csv_train_file, contextual_train_set, self.headers)
        ETLUtils.save_csv_file(
            self.csv_test_file, contextual_test_set, self.headers)

        print('Exported CSV and JSON files: %s'
              % time.strftime("%Y/%d/%m-%H:%M:%S"))

        csv_files = [
            self.csv_train_file,
            self.csv_test_file
        ]

        print('num_cols', len(self.headers))

        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], [], ',', has_header=True,
            suffix='.libfm')

        print('Exported LibFM files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    def predict(self):
        print('predict: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        run_libfm(
            self.context_train_file, self.context_test_file,
            self.context_predictions_file, self.context_log_file)

    def evaluate(self):
        print('evaluate: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        predictions = rmse_calculator.read_targets_from_txt(
            self.context_predictions_file)
        self.top_n_evaluator.evaluate(predictions)
        recall = self.top_n_evaluator.recall

        print('Recall: %f' % recall)

        return recall

    def perform_cross_validation(self):

        self.plant_seeds()

        total_recall = 0.0
        total_cycle_time = 0.0
        num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
        split = 1 - (1/float(num_folds))

        self.create_tmp_file_names()
        self.load()
        self.records = copy.deepcopy(self.original_records)
        if Constants.SHUFFLE_DATA:
            self.shuffle()

        for i in range(0, num_folds):

            cycle_start = time.time()
            cv_start = float(i) / num_folds
            print('\nCycle: %d/%d' % ((i+1), num_folds))

            self.train_records, self.test_records = ETLUtils.split_train_test(
                self.records, split=split, shuffle_data=False, start=cv_start)
            self.export()
            if Constants.USE_CONTEXT:
                lda_based_context = self.train_topic_model()
                self.find_reviews_topics(lda_based_context)
            self.prepare()
            self.predict()
            recall = self.evaluate()
            total_recall += recall

            cycle_end = time.time()
            cycle_time = cycle_end - cycle_start
            total_cycle_time += cycle_time
            print("Total cycle %d time = %f seconds" % ((i+1), cycle_time))

        average_recall = total_recall / num_folds
        average_cycle_time = total_cycle_time / num_folds
        print('average recall: %f' % average_recall)
        print('average cycle time: %f' % average_cycle_time)
        print('End: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        results = copy.deepcopy(Constants._properties)
        results['recall'] = average_recall
        results['cycle_time'] = average_cycle_time

        if not os.path.exists(Constants.RESULTS_FILE):
            with open(Constants.RESULTS_FILE, 'wb') as f:
                w = csv.DictWriter(f, results.keys())
                w.writeheader()
                w.writerow(results)
        else:
            with open(Constants.RESULTS_FILE, 'a') as f:
                w = csv.DictWriter(f, results.keys())
                w.writerow(results)


def run_tests():
    business_type_list = ['hotel']
    split_percentage_list = [80]
    topn_n_list = [10]
    topn_num_items_list = [45]
    lda_alpha_list = [0.005]
    lda_beta_list = [1.0]
    lda_epsilon_list = [0.01]
    # lda_num_topics_list = [50, 150, 450]
    lda_num_topics_list = [50, 150, 450]
    # lda_model_passes_list = [1, 10]
    lda_model_passes_list = [1]
    # lda_model_iterations_list = [50, 500]
    lda_model_iterations_list = [50]
    lda_multicore_list = [False, True]
    cross_validation_num_folds_list = [5]

    combined_properties = combine_parameters(
        business_type_list,
        split_percentage_list,
        topn_n_list,
        topn_num_items_list,
        lda_alpha_list,
        lda_beta_list,
        lda_epsilon_list,
        lda_num_topics_list,
        lda_model_passes_list,
        lda_model_iterations_list,
        lda_multicore_list,
        cross_validation_num_folds_list
    )

    test_cycle = 1
    num_tests = len(combined_properties)
    for properties in combined_properties:
        Constants.update_properties(properties)
        context_top_n_runner = ContextTopNRunner()
        # context_top_n_runner.super_main_lda()

        print('\n\n******************\nTest %d/%d\n******************\n' %
              (test_cycle, num_tests))

        context_top_n_runner.perform_cross_validation()
        test_cycle += 1


def combine_parameters(
        business_type_list,
        split_percentage_list,
        topn_n_list,
        topn_num_items_list,
        lda_alpha_list,
        lda_beta_list,
        lda_epsilon_list,
        lda_num_topics_list,
        lda_model_passes_list,
        lda_model_iterations_list,
        lda_multicore_list,
        cross_validation_num_folds_list
        ):

    combined_properties = []

    for business_type,\
        split_percentage,\
        topn_n,\
        topn_num_items,\
        lda_alpha,\
        lda_beta,\
        lda_epsilon,\
        lda_num_topics,\
        lda_model_passes,\
        lda_model_iterations,\
        lda_multicore,\
        cross_validation_num_folds\
        in itertools.product(
            business_type_list,
            split_percentage_list,
            topn_n_list,
            topn_num_items_list,
            lda_alpha_list,
            lda_beta_list,
            lda_epsilon_list,
            lda_num_topics_list,
            lda_model_passes_list,
            lda_model_iterations_list,
            lda_multicore_list,
            cross_validation_num_folds_list
            ):

        properties = {
            'business_type': business_type,
            'split_percentage': split_percentage,
            'topn_n': topn_n,
            'topn_num_items': topn_num_items,
            'lda_alpha': lda_alpha,
            'lda_beta': lda_beta,
            'lda_epsilon': lda_epsilon,
            'lda_num_topics': lda_num_topics,
            'lda_model_passes': lda_model_passes,
            'lda_model_iterations': lda_model_iterations,
            'lda_multicore': lda_multicore,
            'cross_validation_num_folds': cross_validation_num_folds
        }
        combined_properties.append(properties)

    return combined_properties


def full_cycle(ignore):
    cycle_start = time.time()

    context_top_n_runner = ContextTopNRunner()
    context_top_n_runner.create_tmp_file_names()
    context_top_n_runner.load()
    context_top_n_runner.shuffle()
    context_top_n_runner.split()
    context_top_n_runner.export()
    lda_based_context = context_top_n_runner.train_topic_model()
    context_top_n_runner.find_reviews_topics(lda_based_context)
    context_top_n_runner.prepare()
    context_top_n_runner.predict()
    result = context_top_n_runner.evaluate()
    context_top_n_runner.clear()

    cycle_end = time.time()
    total_cycle_time = cycle_end - cycle_start
    print("Total time = %f seconds" % total_cycle_time)

    return result


def full_cycle_wrapper(args):
    try:
        return full_cycle(args)
    except Exception as e:
        print('Caught exception in worker thread')

        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        raise e


def parallel_context_top_n():

    if not os.path.exists(Constants.USER_ITEM_MAP_FILE):
        records = ETLUtils.load_json_file(Constants.RECORDS_FILE)
        user_item_map = create_user_item_map(records)
        with open(Constants.USER_ITEM_MAP_FILE, 'wb') as write_file:
            pickle.dump(user_item_map, write_file, pickle.HIGHEST_PROTOCOL)

    pool_start_time = time.time()
    pool = Pool()
    print('Total CPUs: %d' % pool._processes)

    num_iterations = Constants.NUM_CYCLES
    # results_list = pool.map(full_cycle_wrapper, range(num_iterations))
    results_list = []
    for i, result in enumerate(
            pool.imap_unordered(full_cycle_wrapper, range(num_iterations)), 1):
        results_list.append(result)
        # sys.stderr.write('\rdone {0:%}'.format(float(i)/num_iterations))
        print('Progress: %2.1f%% (%d/%d)' %
              (float(i)/num_iterations*100, i, num_iterations))
    pool.close()
    pool.join()

    pool_end_time = time.time()
    total_recall = 0.0
    total_pool_time = pool_end_time - pool_start_time
    for recall in results_list:
        total_recall += recall

    average_recall = total_recall / num_iterations
    average_cycle_time = total_pool_time / num_iterations
    print('average recall: %f' % average_recall)
    print('average cycle time: %d seconds' % average_cycle_time)


start = time.time()

my_context_top_n_runner = ContextTopNRunner()
my_context_top_n_runner.perform_cross_validation()
# full_cycle(None)
# run_tests()
# parallel_context_top_n()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
