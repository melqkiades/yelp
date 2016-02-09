from subprocess import call
import time
import cPickle as pickle
from etl import ETLUtils
from etl import libfm_converter
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
from topicmodeling.context.lda_based_context import LdaBasedContext
from tripadvisor.fourcity import extractor
from utils import constants

__author__ = 'fpena'


my_i = 270
SPLIT_PERCENTAGE = '80'
DATASET = 'hotel'
# my_i = 1000
# SPLIT_PERCENTAGE = '98'
# DATASET = 'restaurant'
REVIEW_TYPE = ''
# REVIEW_TYPE = 'specific'
# REVIEW_TYPE = 'generic'

# Folders
DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/stuff/'
LIBFM_FOLDER = '/Users/fpena/tmp/libfm-master/bin/'
GENERATED_FOLDER = DATASET_FOLDER + 'generated_context/'

# Main Files
CACHE_FOLDER = DATASET_FOLDER + 'cache_context/'
RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
               DATASET + 's_shuffled_tagged.json'
# TRAIN_RECORDS_FILE = RECORDS_FILE + '_train'
TRAIN_RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
               DATASET + 's_shuffled_tagged.json_train'
TEST_RECORDS_FILE = RECORDS_FILE + '_test'

# Generated files
RECORDS_TO_PREDICT_FILE = GENERATED_FOLDER +\
                          'records_to_predict_' + DATASET + '.json'
NO_CONTEXT_PREDICTIONS_FILE = GENERATED_FOLDER + 'predictions_' +\
            DATASET + '_no_context.txt'
CONTEXT_PREDICTIONS_FILE =\
            GENERATED_FOLDER + 'predictions_' + DATASET + '_context.txt'

# Cache files
USER_ITEM_MAP_FILE = CACHE_FOLDER + DATASET + '_' + 'user_item_map.pkl'
TOPIC_MODEL_FILE = CACHE_FOLDER + 'topic_model_' + DATASET + '.pkl'


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

    libfm_command = LIBFM_FOLDER + 'libfm'

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
        predictions_file,
        # '-seed',
        # '0'
    ]

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
        self.train_records = None
        self.test_records = None
        self.records_to_predict = None
        self.top_n_evaluator = None
        self.headers = None
        self.important_records = None
        self.context_rich_topics = None

    @staticmethod
    def split():

        print('main split: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        split_command = DATASET_FOLDER + 'split_file.sh'

        command = [
            split_command,
            RECORDS_FILE,
            RECORDS_FILE,
            SPLIT_PERCENTAGE
        ]

        call(command)

    def export(self):
        print('export: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        I = my_i

        self.records = ETLUtils.load_json_file(RECORDS_FILE)
        print('num_records', len(self.records))

        self.test_records = ETLUtils.load_json_file(TEST_RECORDS_FILE)

        if REVIEW_TYPE:
            self.records = ETLUtils.filter_records(
                self.records, constants.PREDICTED_CLASS_FIELD, [REVIEW_TYPE])
            self.test_records = ETLUtils.filter_records(
                self.test_records, constants.PREDICTED_CLASS_FIELD,
                [REVIEW_TYPE])

        user_item_map = create_user_item_map(self.records)

        with open(USER_ITEM_MAP_FILE, 'wb') as write_file:
            pickle.dump(user_item_map, write_file, pickle.HIGHEST_PROTOCOL)

        with open(USER_ITEM_MAP_FILE, 'rb') as read_file:
            user_item_map = pickle.load(read_file)

        self.top_n_evaluator = TopNEvaluator(
            self.records, self.test_records, DATASET, 10, I)
        self.top_n_evaluator.initialize(user_item_map)
        self.top_n_evaluator.export_records_to_predict(RECORDS_TO_PREDICT_FILE)
        self.important_records = self.top_n_evaluator.important_records

    def train_topic_model(self):
        print('train topic model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        self.train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
        lda_based_context = LdaBasedContext(self.train_records)
        lda_based_context.get_context_rich_topics()
        self.context_rich_topics = lda_based_context.context_rich_topics

        print('Trained LDA Model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        #
        # with open(TOPIC_MODEL_FILE, 'wb') as write_file:
        #     pickle.dump(lda_based_context, write_file, pickle.HIGHEST_PROTOCOL)

        # with open(TOPIC_MODEL_FILE, 'rb') as read_file:
        #     lda_based_context = pickle.load(read_file)
        #
        self.context_rich_topics = lda_based_context.context_rich_topics

        return lda_based_context

    def find_reviews_topics(self, lda_based_context):
        print('find topics: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        self.records_to_predict =\
            ETLUtils.load_json_file(RECORDS_TO_PREDICT_FILE)
        lda_based_context.find_contextual_topics(self.train_records)

        topics_map = {}
        lda_based_context.find_contextual_topics(self.important_records)
        for record in self.important_records:
            topics_map[record[constants.REVIEW_ID_FIELD]] =\
                record[constants.TOPICS_FIELD]

        for record in self.records_to_predict:
            topic_distribution = topics_map[record[constants.REVIEW_ID_FIELD]]
            for i in self.context_rich_topics:
                topic_id = 'topic' + str(i[0])
                record[topic_id] = topic_distribution[i[0]]

        print('contextual test set size: %d' % len(self.records_to_predict))

        self.headers = build_headers(self.context_rich_topics)

        print('Exported contextual topics: %s' %
              time.strftime("%Y/%d/%m-%H:%M:%S"))

        return self.train_records, self.records_to_predict

    def prepare(self):
        print('prepare: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        contextual_train_set =\
            ETLUtils.select_fields(self.headers, self.train_records)
        contextual_test_set =\
            ETLUtils.select_fields(self.headers, self.records_to_predict)

        csv_train_file = GENERATED_FOLDER + 'yelp_' + \
            DATASET + '_context_shuffled_train5.csv'
        csv_test_file = GENERATED_FOLDER + 'yelp_' +\
            DATASET + '_context_shuffled_test5.csv'

        ETLUtils.save_csv_file(
            csv_train_file, contextual_train_set, self.headers)
        ETLUtils.save_csv_file(csv_test_file, contextual_test_set, self.headers)

        print('Exported CSV and JSON files: %s'
              % time.strftime("%Y/%d/%m-%H:%M:%S"))

        csv_files = [
            csv_train_file,
            csv_test_file
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

    @staticmethod
    def predict():
        print('predict: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        no_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET +\
            '_context_shuffled_train5.csv.no_context.libfm'
        no_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET +\
            '_context_shuffled_test5.csv.no_context.libfm'

        no_context_log_file = GENERATED_FOLDER + DATASET + '_no_context.log'
        run_libfm(
            no_context_train_file, no_context_test_file,
            NO_CONTEXT_PREDICTIONS_FILE, no_context_log_file)

        context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET +\
            '_context_shuffled_train5.csv.context.libfm'
        context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET +\
            '_context_shuffled_test5.csv.context.libfm'
        context_log_file = GENERATED_FOLDER + DATASET + '_context.log'
        run_libfm(
            context_train_file, context_test_file, CONTEXT_PREDICTIONS_FILE,
            context_log_file)

    def evaluate(self):
        print('evaluate: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        self.top_n_evaluator.load_records_to_predict(RECORDS_TO_PREDICT_FILE)

        predictions =\
            rmse_calculator.read_targets_from_txt(NO_CONTEXT_PREDICTIONS_FILE)
        self.top_n_evaluator.evaluate(predictions)
        no_context_recall = self.top_n_evaluator.recall

        predictions =\
            rmse_calculator.read_targets_from_txt(CONTEXT_PREDICTIONS_FILE)
        self.top_n_evaluator.evaluate(predictions)
        context_recall = self.top_n_evaluator.recall

        print('No context recall: %f' % no_context_recall)
        print('Context recall: %f' % context_recall)

        return context_recall, no_context_recall

    def super_main_lda(self):

        total_context_recall = 0.0
        total_no_context_recall = 0.0
        total_cycle_time = 0.0
        num_iterations = 1

        self.split()
        self.export()

        for i in range(num_iterations):
            cycle_start = time.time()
            print('\nCycle: %d' % i)

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
            print("Total cycle %d time = %f seconds" % (i, cycle_time))

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

start = time.time()
# main()
# split()
# export()
# main_lda()
# predict()
# evaluate()
# super_main_lda()
# experiment()
# split_binary_reviews()
context_top_n_runner = ContextTopNRunner()
context_top_n_runner.super_main_lda()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
