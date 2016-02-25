import copy
from multiprocessing import Pool
import os
import random
from subprocess import call
import time
import cPickle as pickle
import traceback
import uuid
import numpy
from etl import ETLUtils
from etl import libfm_converter
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
from topicmodeling.context.lda_based_context import LdaBasedContext
from tripadvisor.fourcity import extractor
from utils import constants

__author__ = 'fpena'


def build_headers(context_rich_topics):
    headers = [
        constants.RATING_FIELD,
        constants.USER_ID_FIELD,
        constants.ITEM_ID_FIELD
    ]
    for topic in context_rich_topics:
        topic_id = 'topic' + str(topic[0])
        headers.append(topic_id)
    return headers


def create_user_item_map(records):
    user_ids = extractor.get_groupby_list(records, constants.USER_ID_FIELD)
    user_item_map = {}
    user_count = 0

    for user_id in user_ids:
        user_records =\
            ETLUtils.filter_records(records, constants.USER_ID_FIELD, [user_id])
        user_items =\
            extractor.get_groupby_list(user_records, constants.ITEM_ID_FIELD)
        user_item_map[user_id] = user_items
        user_count += 1

        # print("user count %d" % user_count),
        print 'user count: {0}\r'.format(user_count),

    print

    return user_item_map


def run_libfm(train_file, test_file, predictions_file, log_file):

    libfm_command = constants.LIBFM_FOLDER + 'libFM'

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

    if constants.LIBFM_SEED is not None:
        command.extend(['-seed', str(constants.LIBFM_SEED)])

    f = open(log_file, "w")
    call(command, stdout=f)


def filter_reviews(records, reviews, review_type):
    print('filter: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    if not review_type:
        return records, reviews

    filtered_records = []
    filtered_reviews = []

    for record, review in zip(records, reviews):
        if record[constants.PREDICTED_CLASS_FIELD] == review_type:
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
        self.context_rich_topics = None
        self.csv_train_file = None
        self.csv_test_file = None
        self.context_predictions_file = None
        self.context_train_file = None
        self.context_test_file = None
        self.context_log_file = None
        self.no_context_predictions_file = None
        self.no_context_train_file = None
        self.no_context_test_file = None
        self.no_context_log_file = None

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
        os.remove(self.no_context_predictions_file)
        os.remove(self.no_context_train_file)
        os.remove(self.no_context_test_file)
        os.remove(self.no_context_log_file)

        self.csv_train_file = None
        self.csv_test_file = None
        self.context_predictions_file = None
        self.context_train_file = None
        self.context_test_file = None
        self.context_log_file = None
        self.no_context_predictions_file = None
        self.no_context_train_file = None
        self.no_context_test_file = None
        self.no_context_log_file = None

    def create_tmp_file_names(self):

        unique_id = uuid.uuid4().hex
        prefix = constants.GENERATED_FOLDER + unique_id + '_' +\
            constants.ITEM_TYPE
        # prefix = constants.GENERATED_FOLDER + constants.ITEM_TYPE

        # print('unique id: %s' % unique_id)
        
        self.csv_train_file = prefix + '_context_train.csv'
        self.csv_test_file = prefix + '_context_test.csv'
        self.context_predictions_file = prefix + '_context_predictions.txt'
        self.context_train_file = self.csv_train_file + '.context.libfm'
        self.context_test_file = self.csv_test_file + '.context.libfm'
        self.context_log_file = prefix + '_context.log'
        self.no_context_predictions_file =\
            prefix + '_no_context_predictions.txt'
        self.no_context_train_file = self.csv_train_file + '.no_context.libfm'
        self.no_context_test_file = self.csv_test_file + '.no_context.libfm'
        self.no_context_log_file = prefix + '_no_context.log'

    def load(self):
        print('load: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        self.original_records = ETLUtils.load_json_file(constants.RECORDS_FILE)
        print('num_records', len(self.original_records))

        if not os.path.exists(constants.USER_ITEM_MAP_FILE):
            records = ETLUtils.load_json_file(constants.RECORDS_FILE)
            user_item_map = create_user_item_map(records)
            with open(constants.USER_ITEM_MAP_FILE, 'wb') as write_file:
                pickle.dump(user_item_map, write_file, pickle.HIGHEST_PROTOCOL)

    def shuffle(self):
        print('shuffle: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        random.shuffle(self.records)

    def split(self):
        print('split: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        num_records = len(self.records)
        num_split_records =\
            int(float(constants.SPLIT_PERCENTAGE)/100*num_records)
        self.train_records = self.records[:num_split_records]
        self.test_records = self.records[num_split_records:]

    def export(self):
        print('export: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        if constants.REVIEW_TYPE:
            self.records = ETLUtils.filter_records(
                self.records, constants.PREDICTED_CLASS_FIELD,
                [constants.REVIEW_TYPE])
            self.test_records = ETLUtils.filter_records(
                self.test_records, constants.PREDICTED_CLASS_FIELD,
                [constants.REVIEW_TYPE])

        with open(constants.USER_ITEM_MAP_FILE, 'rb') as read_file:
            user_item_map = pickle.load(read_file)

        self.top_n_evaluator = TopNEvaluator(
            self.records, self.test_records, constants.ITEM_TYPE, 10,
            constants.TOPN_NUM_ITEMS)
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
            topics_map[record[constants.REVIEW_ID_FIELD]] =\
                record[constants.TOPICS_FIELD]

        for record in self.records_to_predict:
            topic_distribution = topics_map[record[constants.REVIEW_ID_FIELD]]

            context_topics = {}
            for i in self.context_rich_topics:
                topic_id = 'topic' + str(i[0])
                context_topics[topic_id] = topic_distribution[i[0]]

            record[constants.CONTEXT_TOPICS_FIELD] = context_topics

        print('contextual test set size: %d' % len(self.records_to_predict))

        self.headers = build_headers(self.context_rich_topics)

        print('Exported contextual topics: %s' %
              time.strftime("%Y/%d/%m-%H:%M:%S"))

        return self.train_records, self.records_to_predict

    def prepare(self):
        print('prepare: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        contextual_train_set = copy.deepcopy(self.train_records)
        for record in contextual_train_set:
            record.update(record[constants.CONTEXT_TOPICS_FIELD])
        contextual_train_set = ETLUtils.select_fields(self.headers, contextual_train_set)

        contextual_test_set = copy.deepcopy(self.records_to_predict)
        for record in contextual_test_set:
            record.update(record[constants.CONTEXT_TOPICS_FIELD])
        contextual_test_set = ETLUtils.select_fields(self.headers, contextual_test_set)

        ETLUtils.drop_fields([constants.TOPICS_FIELD], self.train_records)
        # ETLUtils.drop_fields([constants.TOPICS_FIELD], self.records_to_predict)

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

        num_cols = len(self.headers)
        context_cols = num_cols
        print('num_cols', num_cols)
        # print('context_cols', context_cols)

        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], range(3, context_cols), ',', has_header=True,
            suffix='.no_context.libfm')
        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], [], ',', has_header=True,
            suffix='.context.libfm')

        print('Exported LibFM files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    def predict(self):
        print('predict: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        run_libfm(
            self.no_context_train_file, self.no_context_test_file,
            self.no_context_predictions_file, self.no_context_log_file)
        run_libfm(
            self.context_train_file, self.context_test_file,
            self.context_predictions_file, self.context_log_file)

    def evaluate(self):
        print('evaluate: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        predictions = rmse_calculator.read_targets_from_txt(
            self.no_context_predictions_file)
        self.top_n_evaluator.evaluate(predictions)
        no_context_recall = self.top_n_evaluator.recall

        predictions = rmse_calculator.read_targets_from_txt(
            self.context_predictions_file)
        self.top_n_evaluator.evaluate(predictions)
        context_recall = self.top_n_evaluator.recall

        print('No context recall: %f' % no_context_recall)
        print('Context recall: %f' % context_recall)

        return context_recall, no_context_recall

    def super_main_lda(self):

        total_context_recall = 0.0
        total_no_context_recall = 0.0
        total_cycle_time = 0.0
        num_iterations = constants.NUM_CYCLES

        context_top_n_runner.create_tmp_file_names()
        self.load()

        for i in range(num_iterations):

            cycle_start = time.time()
            print('\nCycle: %d/%d' % ((i+1), num_iterations))

            self.records = copy.deepcopy(self.original_records)
            self.shuffle()
            self.split()
            self.export()
            lda_based_context = self.train_topic_model()
            self.find_reviews_topics(lda_based_context)
            self.prepare()
            self.predict()
            context_recall, no_context_recall = self.evaluate()
            total_context_recall += context_recall
            total_no_context_recall += no_context_recall

            cycle_end = time.time()
            cycle_time = cycle_end - cycle_start
            total_cycle_time += cycle_time
            print("Total cycle %d time = %f seconds" % ((i+1), cycle_time))

        average_context_recall = total_context_recall / num_iterations
        average_no_context_recall = total_no_context_recall / num_iterations
        average_cycle_time = total_cycle_time / num_iterations
        improvement =\
            (average_context_recall / average_no_context_recall - 1) * 100
        print('average no context recall', average_no_context_recall)
        print('average context recall', average_context_recall)
        print('average improvement: %f2.3%%' % improvement)
        print('average cycle time', average_cycle_time)
        print('End: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))


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

    if not os.path.exists(constants.USER_ITEM_MAP_FILE):
        records = ETLUtils.load_json_file(constants.RECORDS_FILE)
        user_item_map = create_user_item_map(records)
        with open(constants.USER_ITEM_MAP_FILE, 'wb') as write_file:
            pickle.dump(user_item_map, write_file, pickle.HIGHEST_PROTOCOL)

    pool_start_time = time.time()
    pool = Pool()
    print('Total CPUs: %d' % pool._processes)

    num_iterations = constants.NUM_CYCLES
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
    total_context_recall = 0.0
    total_no_context_recall = 0.0
    total_pool_time = pool_end_time - pool_start_time
    for context_recall, no_context_recall in results_list:
        total_context_recall += context_recall
        total_no_context_recall += no_context_recall

    average_context_recall = total_context_recall / num_iterations
    average_no_context_recall = total_no_context_recall / num_iterations
    average_cycle_time = total_pool_time / num_iterations
    improvement =\
        (average_context_recall / average_no_context_recall - 1) * 100
    print('average no context recall: %f' % average_no_context_recall)
    print('average context recall: %f' % average_context_recall)
    print('average improvement: %f%%' % improvement)
    print('average cycle time: %d seconds' % average_cycle_time)


start = time.time()
if constants.RANDOM_SEED is not None:
    print('random seed: %d' % constants.RANDOM_SEED)
    random.seed(constants.RANDOM_SEED)
if constants.NUMPY_RANDOM_SEED is not None:
    print('numpy random seed: %d' % constants.NUMPY_RANDOM_SEED)
    numpy.random.seed(constants.NUMPY_RANDOM_SEED)

context_top_n_runner = ContextTopNRunner()
context_top_n_runner.super_main_lda()
# full_cycle(None)
# parallel_context_top_n()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
