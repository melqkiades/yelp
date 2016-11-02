import argparse
import os
import random
import time
import traceback
from multiprocessing import Pool
import cPickle as pickle

from os.path import expanduser

from etl import ETLUtils
from topicmodeling.context.lda_based_context import LdaBasedContext
from topicmodeling.context.nmf_context_extractor import NmfContextExtractor
from utils import constants
from utils import utilities
from utils.constants import Constants


def create_topic_model(records, cycle_index, fold_index, check_exists=True):

    print('%s: Create topic model' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    topic_model_file_path = \
        Constants.generate_file_name(
            'topic_model', 'pkl', Constants.CACHE_FOLDER,
            cycle_index, fold_index, True)

    print(topic_model_file_path)

    if check_exists and os.path.exists(topic_model_file_path):
        print('WARNING: Topic model already exists')
        return load_topic_model(cycle_index, fold_index)

    topic_model = train_context_extractor(records)

    with open(topic_model_file_path, 'wb') as write_file:
        pickle.dump(topic_model, write_file, pickle.HIGHEST_PROTOCOL)

    return topic_model


def train_context_extractor(records):
    print('%s: train context topics model' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    if Constants.TOPIC_MODEL_TYPE == 'lda':
        context_extractor = LdaBasedContext(records)
        context_extractor.generate_review_corpus()
        context_extractor.build_topic_model()
        context_extractor.update_reviews_with_topics()
        context_extractor.get_context_rich_topics()
    elif Constants.TOPIC_MODEL_TYPE == 'nmf':
        context_extractor = NmfContextExtractor(records)
        context_extractor.generate_review_bows()
        context_extractor.build_document_term_matrix()
        context_extractor.build_stable_topic_model()
        context_extractor.update_reviews_with_topics()
        context_extractor.get_context_rich_topics()
    else:
        raise ValueError('Unrecognized topic model type')

    print('%s: Trained Topic Model' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    return context_extractor


def load_topic_model(cycle_index, fold_index):
    file_path = \
        Constants.generate_file_name(
            'topic_model', 'pkl', Constants.CACHE_FOLDER,
            cycle_index, fold_index, True)
    print(file_path)
    with open(file_path, 'rb') as read_file:
        topic_model = pickle.load(read_file)
    return topic_model


def create_single_topic_model(cycle_index, fold_index, check_exists=True):

    Constants.print_properties()
    print('%s: Start' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
        msg = 'This function shouldn\'t be used when the ' \
              'separate_topic_model_recsys_reviews property is set to True'
        raise ValueError(msg)

    records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)

    if Constants.CROSS_VALIDATION_STRATEGY == 'nested_test':
        pass
    elif Constants.CROSS_VALIDATION_STRATEGY == 'nested_validate':
        num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
        cycle = Constants.NESTED_CROSS_VALIDATION_CYCLE
        split = 1 - (1 / float(num_folds))
        cv_start = float(cycle) / num_folds
        print('cv_start', cv_start)
        records, _ = ETLUtils.split_train_test(records, split, cv_start)
    else:
        raise ValueError('Unknown cross-validation strategy')

    utilities.plant_seeds()
    num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
    split = 1 - (1/float(num_folds))

    for i in range(cycle_index+1):

        if Constants.SHUFFLE_DATA:
            random.shuffle(records)

    cv_start = float(fold_index) / num_folds
    train_records, test_records = \
        ETLUtils.split_train_test(records, split=split, start=cv_start)
    return create_topic_model(train_records, cycle_index, fold_index, check_exists)


def create_topic_models():

    Constants.print_properties()
    print('%s: Start' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    records = ETLUtils.load_json_file(Constants.RECORDS_FILE)

    utilities.plant_seeds()
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

        # lda_based_context = train_context_extractor(train_records)
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
    print('%s: Generating file with commands' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))

    code_path = constants.CODE_FOLDER[:-1]
    python_path = "PYTHONPATH='" + code_path + "' "
    python_command = "stdbuf -oL nohup python "
    base_file_name = "topicmodel-" + Constants.ITEM_TYPE
    log_file = " > ~/logs/" + base_file_name + "-%d-%d.log"
    commands_dir = expanduser("~") + "/tmp/"
    commands_file = commands_dir + base_file_name + ".sh"
    command_list = []

    for cycle in range(Constants.NUM_CYCLES):
        for fold in range(Constants.CROSS_VALIDATION_NUM_FOLDS):
            python_file = constants.CODE_FOLDER +\
                "topicmodeling/context/topic_model_creator.py -c %d -f %d"
            full_command = python_path + python_command + python_file + log_file
            command_list.append(full_command % (cycle, fold, cycle+1, fold+1))

    with open(commands_file, "w") as write_file:
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
# create_single_topic_model(None, None)
# num_cycles = Constants.NUM_CYCLES
# num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
# for cycle, fold in itertools.product(range(num_cycles), range(num_folds)):
#     create_single_topic_model(cycle, fold)
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

# if __name__ == '__main__':
#     start = time.time()
#     main()
#     end = time.time()
#     total_time = end - start
#     print("Total time = %f seconds" % total_time)
#
# generate_file_with_commands()
