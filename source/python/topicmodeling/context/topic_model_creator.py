import cPickle as pickle
import os
import random
import time
import traceback
from multiprocessing import Pool

import numpy

from etl import ETLUtils
from topicmodeling.context.lda_based_context import LdaBasedContext
from utils.constants import Constants


def get_topic_model_file_path(cycle_index, fold_index):

    if Constants.REVIEW_TYPE == Constants.ALL_TOPICS:
        all_topics = 'alltopics_'
    else:
        all_topics = ''

    topic_model_file = Constants.ITEM_TYPE + '_' + all_topics +\
                       'topic_model_cycle:' +\
                       str(cycle_index+1) + '|' + str(Constants.NUM_CYCLES) +\
                       '_fold:' + str(fold_index+1) + '|' +\
                       str(Constants.CROSS_VALIDATION_NUM_FOLDS) +\
                       '_numtopics:' + str(Constants.LDA_NUM_TOPICS) +\
                       '_iterations:' + str(Constants.LDA_MODEL_ITERATIONS) +\
                       '_passes:' + str(Constants.LDA_MODEL_PASSES) +\
                       '.pkl'
    return Constants.CACHE_FOLDER + topic_model_file


def create_topic_model(records, cycle_index, fold_index):

    topic_model_file_path = get_topic_model_file_path(cycle_index, fold_index)

    print(topic_model_file_path)

    if not os.path.exists(topic_model_file_path):
        topic_model = train_topic_model(records)
        with open(topic_model_file_path, 'wb') as write_file:
            pickle.dump(topic_model, write_file, pickle.HIGHEST_PROTOCOL)


def plant_seeds():

    if Constants.RANDOM_SEED is not None:
        print('random seed: %d' % Constants.RANDOM_SEED)
        random.seed(Constants.RANDOM_SEED)
    if Constants.NUMPY_RANDOM_SEED is not None:
        print('numpy random seed: %d' % Constants.NUMPY_RANDOM_SEED)
        numpy.random.seed(Constants.NUMPY_RANDOM_SEED)


def train_topic_model(records):
    print('train topic model: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    lda_based_context = LdaBasedContext(records)
    lda_based_context.get_context_rich_topics()
    print('Trained LDA Model: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    return lda_based_context


def load_topic_model(cycle_index, fold_index):
    file_path = get_topic_model_file_path(cycle_index, fold_index)
    with open(file_path, 'rb') as read_file:
        topic_model = pickle.load(read_file)
    return topic_model


def create_topic_models():

    print(Constants._properties)

    records = ETLUtils.load_json_file(Constants.RECORDS_FILE)

    plant_seeds()
    num_cycles = Constants.NUM_CYCLES
    num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
    split = 1 - (1/float(num_folds))

    for i in range(num_cycles):

        print('\n\nCycle: %d/%d' % ((i+1), num_cycles))

        if Constants.SHUFFLE_DATA:
            random.shuffle(records)

        train_records_list = []

        for j in range(num_folds):

            cv_start = float(j) / num_folds

            train_records, test_records =\
                ETLUtils.split_train_test(records, split=split, start=cv_start)
            train_records_list.append(train_records)

        args = zip(
            train_records_list,
            [i] * Constants.CROSS_VALIDATION_NUM_FOLDS,
            range(Constants.CROSS_VALIDATION_NUM_FOLDS)
        )

        parallel_context_top_n(args)

        # lda_based_context = train_topic_model(train_records)
        # create_topic_model(i+1, j+1)


def create_topic_model_wrapper(args):
    try:
        return create_topic_model(*args)
    except Exception as e:
        print('Caught exception in worker thread')

        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        raise e


def parallel_context_top_n(args):

    pool_start_time = time.time()
    pool = Pool()
    print('Total CPUs: %d' % pool._processes)

    num_iterations = len(args)
    # results_list = pool.map(full_cycle_wrapper, range(num_iterations))
    results_list = []
    for i, result in enumerate(
            pool.imap_unordered(create_topic_model_wrapper, args),
            1):
        results_list.append(result)
        # sys.stderr.write('\rdone {0:%}'.format(float(i)/num_iterations))
        print('Progress: %2.1f%% (%d/%d)' %
              (float(i)/num_iterations*100, i, num_iterations))
    pool.close()
    pool.join()

    pool_end_time = time.time()
    total_pool_time = pool_end_time - pool_start_time

    average_cycle_time = total_pool_time / num_iterations
    print('average cycle time: %d seconds' % average_cycle_time)


def main():
    create_topic_models()
    # parallel_context_top_n()
    # test()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

