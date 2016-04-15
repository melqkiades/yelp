import argparse
import os
import random
import time
import traceback
from multiprocessing import Pool
import cPickle as pickle
import numpy
from os.path import expanduser

from etl import ETLUtils
from topicmodeling.context.lda_based_context import LdaBasedContext
from utils import constants
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
                       '-3.pkl'
    return Constants.CACHE_FOLDER + topic_model_file


def create_topic_model(records, cycle_index, fold_index):

    print('%s: Create topic model' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    topic_model_file_path = get_topic_model_file_path(cycle_index, fold_index)

    print(topic_model_file_path)

    if os.path.exists(topic_model_file_path):
        print('topic model already exists')
        return

    if Constants.REVIEW_TYPE == Constants.ALL_TOPICS:
        topic_model = train_all_topics_model(records)
    else:
        topic_model = train_context_topics_model(records)

    with open(topic_model_file_path, 'wb') as write_file:
        pickle.dump(topic_model, write_file, pickle.HIGHEST_PROTOCOL)


def plant_seeds():

    if Constants.RANDOM_SEED is not None:
        print('random seed: %d' % Constants.RANDOM_SEED)
        random.seed(Constants.RANDOM_SEED)
    if Constants.NUMPY_RANDOM_SEED is not None:
        print('numpy random seed: %d' % Constants.NUMPY_RANDOM_SEED)
        numpy.random.seed(Constants.NUMPY_RANDOM_SEED)


def train_context_topics_model(records):
    print('%s: train context topics model' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    lda_based_context = LdaBasedContext(records)
    lda_based_context.generate_review_corpus()
    lda_based_context.build_topic_model()
    lda_based_context.update_reviews_with_topics()

    print('%s: Trained LDA Model' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    return lda_based_context


def train_all_topics_model(records):
    print('%s: train all topics model' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    lda_based_context = LdaBasedContext(records)
    lda_based_context.get_all_topics()

    print('%s: Trained LDA Model' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    return lda_based_context


def load_topic_model(cycle_index, fold_index):
    file_path = get_topic_model_file_path(cycle_index, fold_index)
    with open(file_path, 'rb') as read_file:
        topic_model = pickle.load(read_file)
    return topic_model


def create_single_topic_model(cycle_index, fold_index):

    print(Constants._properties)
    print('%s: Start' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    records = ETLUtils.load_json_file(Constants.RECORDS_FILE)

    plant_seeds()
    num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
    split = 1 - (1/float(num_folds))

    for i in range(cycle_index+1):

        if Constants.SHUFFLE_DATA:
            random.shuffle(records)

    cv_start = float(fold_index) / num_folds
    train_records, test_records = \
        ETLUtils.split_train_test(records, split=split, start=cv_start)
    create_topic_model(train_records, cycle_index, fold_index)


def create_topic_models():

    print(Constants._properties)
    print('%s: Start' % time.strftime("%Y/%m/%d-%H:%M:%S"))

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

        # lda_based_context = train_context_topics_model(train_records)
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
    pool = Pool(2)
    print('Total CPUs: %d' % pool._processes)

    num_iterations = len(args)
    # results_list = pool.map(full_cycle_wrapper, range(num_iterations))
    results_list = []
    for i, result in enumerate(
            pool.imap_unordered(create_topic_model_wrapper, args),
            1):
        results_list.append(result)
        # sys.stderr.write('\rdone {0:%}'.format(float(i)/num_iterations))
        print('%s: Progress: %2.1f%% (%d/%d)' %
              (time.strftime("%Y/%m/%d-%H:%M:%S"),
               float(i)/num_iterations*100, i, num_iterations))
    pool.close()
    pool.join()

    pool_end_time = time.time()
    total_pool_time = pool_end_time - pool_start_time

    average_cycle_time = total_pool_time / num_iterations
    print('average cycle time: %d seconds' % average_cycle_time)


def generate_file_with_commands():

    code_path = constants.CODE_FOLDER[:-1]
    python_path = "PYTHONPATH='" + code_path + "' "
    python_command = "stdbuf -oL nohup python "
    review_type = ''
    if Constants.ALL_TOPICS is not None:
        review_type = Constants.ALL_TOPICS + "-"
    log_file = " > ~/logs/" "topicmodel-" + review_type + Constants.ITEM_TYPE +\
               "-%d-%d.log"
    command_list = []

    for cycle in range(Constants.NUM_CYCLES):
        for fold in range(Constants.CROSS_VALIDATION_NUM_FOLDS):
            python_file = constants.CODE_FOLDER +\
                "topicmodeling/context/topic_model_creator.py -c %d -f %d"
            full_command = python_path + python_command + python_file + log_file
            command_list.append(full_command % (cycle, fold, cycle+1, fold+1))

    home = expanduser("~")
    with open(home + "/tmp/command_list.txt", "w") as write_file:
        for command in command_list:
            write_file.write("%s\n" % command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--cycle', metavar='int', type=int,
        nargs=1, help='The index of the running cycle')
    parser.add_argument(
        '-f', '--fold', metavar='int', type=int,
        nargs=1, help='The index of the cross validation fold')

    args = parser.parse_args()
    fold = args.fold[0]
    cycle = args.cycle[0]
    create_single_topic_model(cycle, fold)

# start = time.time()
# create_single_topic_models(0, 0)
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
#
# if __name__ == '__main__':
#     start = time.time()
#     main()
#     end = time.time()
#     total_time = end - start
#     print("Total time = %f seconds" % total_time)
#
# generate_file_with_commands()
