from subprocess import call
import time
import cPickle as pickle

from etl import ETLUtils
from etl.libfm_converter import csv_to_libfm
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator

__author__ = 'fpena'


my_i = 270
SPLIT_PERCENTAGE = '80'
DATASET = 'hotel'
# my_i = 1000
# SPLIT_PERCENTAGE = '98'
# DATASET = 'restaurant'
REVIEW_TYPE = ''
# REVIEW_TYPE = 'specific_'
# REVIEW_TYPE = 'generic_'

DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/'
LIBFM_FOLDER = '/Users/fpena/tmp/libfm-1.42.src/bin/'
GENERATED_FOLDER = DATASET_FOLDER + 'generated_plain/'
RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' + DATASET + 's_shuffled_tagged.json'
TRAIN_RECORDS_FILE = RECORDS_FILE + '_train'
TEST_RECORDS_FILE = RECORDS_FILE + '_test'
RECORDS_TO_PREDICT_FILE = GENERATED_FOLDER + 'records_to_predict_' + DATASET + '.json'

CACHE_FOLDER = DATASET_FOLDER + 'cache_plain/'
USER_ITEM_MAP_FILE = CACHE_FOLDER + DATASET + '_user_item_map.pkl'


def main_split():

    split_command = DATASET_FOLDER + 'split_file.sh'

    command = [
        split_command,
        RECORDS_FILE,
        RECORDS_FILE,
        SPLIT_PERCENTAGE
    ]

    call(command)


def main_export():
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    print('num_records', len(records))

    test_records = ETLUtils.load_json_file(TEST_RECORDS_FILE)
    # test_reviews = review_metrics_extractor.build_reviews(test_records)
    # with open(TEST_REVIEWS_FILE, 'wb') as write_file:
    #     pickle.dump(test_reviews, write_file, pickle.HIGHEST_PROTOCOL)
    # with open(TEST_REVIEWS_FILE, 'rb') as read_file:
    #     test_reviews = pickle.load(read_file)
    # train_file = RECORDS_FILE + '_train'
    # train_records = ETLUtils.load_json_file(train_file)

    with open(USER_ITEM_MAP_FILE, 'rb') as read_file:
        user_item_map = pickle.load(read_file)

    top_n_evaluator = TopNEvaluator(records, test_records, DATASET, 10, I)
    top_n_evaluator.initialize(user_item_map)

    top_n_evaluator.export_records_to_predict(RECORDS_TO_PREDICT_FILE)


def main_evaluate():
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    # print('num_records', len(records))

    test_file = RECORDS_FILE + '_test'
    test_records = ETLUtils.load_json_file(test_file)

    top_n_evaluator = TopNEvaluator(records, test_records, DATASET, 10, I)
    top_n_evaluator.find_important_records()
    # top_n_evaluator.initialize()

    # records_to_predict_file = DATASET_FOLDER + 'generated/records_to_predict_' + DATASET + '.json'
    top_n_evaluator.load_records_to_predict(RECORDS_TO_PREDICT_FILE)

    predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)

    # print('total predictions', len(predictions))
    top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    print('recall', top_n_evaluator.recall)

    return top_n_evaluator.recall


def main_converter():

    csv_train_file = GENERATED_FOLDER + 'yelp_training_set_review_' + DATASET + 's_shuffled_train.csv'
    csv_test_file = GENERATED_FOLDER + 'records_to_predict_' + DATASET + '.csv'

    # ETLUtils.json_to_csv(TRAIN_RECORDS_FILE, csv_train_file, 'user_id', 'business_id', 'stars', False, True)
    # ETLUtils.json_to_csv(RECORDS_TO_PREDICT_FILE, csv_test_file, 'user_id', 'business_id', 'stars', False, True)

    headers = ['stars', 'user_id', 'business_id']
    train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
    records_to_predict = ETLUtils.load_json_file(RECORDS_TO_PREDICT_FILE)
    train_records = ETLUtils.select_fields(headers, train_records)
    records_to_predict = ETLUtils.select_fields(headers, records_to_predict)

    ETLUtils.save_csv_file(csv_train_file, train_records, headers)
    ETLUtils.save_csv_file(csv_test_file, records_to_predict, headers)

    csv_files = [
        csv_train_file,
        csv_test_file
    ]

    csv_to_libfm(csv_files, 0, [1, 2], [], ',', has_header=True)


def main_libfm():

    train_file = GENERATED_FOLDER + 'yelp_training_set_review_' + DATASET + 's_shuffled_train.csv.libfm'
    test_file = GENERATED_FOLDER + 'records_to_predict_' + DATASET + '.csv.libfm'
    predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '.txt'
    log_file = GENERATED_FOLDER + DATASET + '.log'

    run_libfm(train_file, test_file, predictions_file, log_file)


def super_main():

    total_recall = 0.0
    num_iterations = 10

    for i in range(num_iterations):
        print('\nCycle: %d' % i)

        print('main split')
        main_split()
        print('main export')
        main_export()
        print('main converter')
        main_converter()
        print('main libfm')
        main_libfm()
        print('main evaluate')
        total_recall += main_evaluate()

    average_recall = total_recall / num_iterations
    print('average_recall', average_recall)


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
        predictions_file
    ]

    f = open(log_file, "w")
    call(command, stdout=f)





start = time.time()
# main()
# main_split()
# main_export()
# main_converter()
# main_libfm()
# main_evaluate()
super_main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)

