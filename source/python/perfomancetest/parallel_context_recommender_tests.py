# import sys
# sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')

import traceback
from etl import ETLUtils
from evaluation import precision_in_top_n
from perfomancetest.context_recommender_tests import RMSE_HEADERS
from perfomancetest.context_recommender_tests import TOPN_HEADERS
import time
from multiprocessing import Pool
import itertools
import cPickle as pickle
from perfomancetest import context_recommender_tests
from tripadvisor.fourcity import recommender_evaluator
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


# This wrapper is necessary when you want to pass a function more
# than 1 parameter
def run_rmse_test_wrapper(args):
    try:
        return recommender_evaluator.perform_cross_validation(*args)
    except Exception as e:
        print('Caught exception in worker thread')

        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        raise e


# This wrapper is necessary when you want to pass a function more
# than 1 parameter
def run_topn_test_wrapper(args):
    try:
        return precision_in_top_n.calculate_recall_in_top_n(*args)
    except Exception as e:
        print('Caught exception in worker thread')

        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        raise e


def parallel_run_rmse_test(
        records_file, recommenders, binary_reviews_file, reviews_type=None):

    records = context_recommender_tests.load_records(records_file)
    records = extractor.remove_users_with_low_reviews(records, 20)
    with open(binary_reviews_file, 'rb') as read_file:
        binary_reviews = pickle.load(read_file)

    if len(records) != len(binary_reviews):
        raise ValueError("The records and reviews should have the same length")

    num_folds = 5

    args = itertools.product(
        [records],
        recommenders,
        [num_folds],
        [binary_reviews],
        [reviews_type]
    )

    print('Total recommenders: %d' % (len(recommenders)))

    pool = Pool()

    print('Total CPUs: %d' % pool._processes)

    results_list = pool.map(run_rmse_test_wrapper, args)
    pool.close()
    pool.join()

    # After we have finished executing, we process the results
    dataset_info_map = {}
    dataset_info_map['dataset'] = records_file.split('/')[-1]
    dataset_info_map['cache_reviews'] = binary_reviews_file.split('/')[-1]
    dataset_info_map['num_records'] = len(records)
    dataset_info_map['reviews_type'] = reviews_type
    dataset_info_map['cross_validation_folds'] = num_folds

    results_log_list = []
    for recommender, results in zip(recommenders, results_list):
        results_log_list.append(context_recommender_tests.process_rmse_results(
            recommender, results, dataset_info_map))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'recommender-rmse-results-parallel' + timestamp

    ETLUtils.save_csv_file(file_name + '.csv', results_log_list, RMSE_HEADERS, '\t')

    return results_list


def parallel_run_topn_test(
        records_file, recommenders, binary_reviews_file, reviews_type=None):

    records = context_recommender_tests.load_records(records_file)
    records = extractor.remove_users_with_low_reviews(records, 20)
    with open(binary_reviews_file, 'rb') as read_file:
        binary_reviews = pickle.load(read_file)

    if len(records) != len(binary_reviews):
        raise ValueError("The records and reviews should have the same length")

    num_folds = 5
    split = 0.986
    top_n = 10
    min_like_score = 5.0

    args = itertools.product(
        [records],
        recommenders,
        [top_n],
        [num_folds],
        [split],
        [min_like_score],
        [binary_reviews],
        [reviews_type]
    )

    print('Total recommenders: %d' % (len(recommenders)))

    pool = Pool()

    print('Total CPUs: %d' % pool._processes)

    results_list = pool.map(run_topn_test_wrapper, args)
    pool.close()
    pool.join()

    # After we have finished executing, we process the results
    dataset_info_map = {}
    dataset_info_map['dataset'] = records_file.split('/')[-1]
    dataset_info_map['cache_reviews'] = binary_reviews_file.split('/')[-1]
    dataset_info_map['num_records'] = len(records)
    dataset_info_map['reviews_type'] = reviews_type
    dataset_info_map['cross_validation_folds'] = num_folds
    dataset_info_map['min_like_score'] = min_like_score
    dataset_info_map['top_n'] = top_n

    results_log_list = []
    for recommender, results in zip(recommenders, results_list):
        results_log_list.append(context_recommender_tests.process_topn_results(
            recommender, results, dataset_info_map))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'recommender-topn-results-parallel' + timestamp

    ETLUtils.save_csv_file(file_name + '.csv', results_log_list, TOPN_HEADERS, '\t')

    return results_list


def main():
    print('Process start: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    my_records_file = folder + 'yelp_training_set_review_restaurants_shuffled.json'
    # my_records_file = folder + 'yelp_training_set_review_hotels_shuffled.json'
    # my_records_file = folder + 'yelp_training_set_review_spas_shuffled.json'
    my_binary_reviews_file = folder + 'reviews_restaurant_shuffled_20.pkl'
    # my_binary_reviews_file = folder + 'reviews_hotel_shuffled.pkl'
    # my_binary_reviews_file = folder + 'reviews_spa_shuffled_2.pkl'
    # my_binary_reviews_file = folder + 'reviews_context_hotel_2.pkl'

    combined_recommenders = context_recommender_tests.get_recommenders_set()

    # run_rmse_test(my_records_file, combined_recommenders, my_binary_reviews_file)
    parallel_run_rmse_test(
        my_records_file, combined_recommenders, my_binary_reviews_file)
    # parallel_run_topn_test(
    #     my_records_file, combined_recommenders, my_binary_reviews_file)



start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
