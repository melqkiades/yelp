import argparse
import codecs
import os
import random
import time
import cPickle as pickle

from os.path import expanduser

import shutil
from gensim import corpora
from gensim.models import ldamodel

from etl import ETLUtils
from topicmodeling import topic_ensemble_caller
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


def train_topic_model(records):
    print('%s: train topic model' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    if Constants.TOPIC_MODEL_TYPE == 'lda':

        topic_model_file_path = \
            Constants.generate_file_name(
                'topic_model', 'pkl', Constants.CACHE_FOLDER, None, None, True)
        if os.path.exists(topic_model_file_path):
            print('WARNING: Topic model already exists')
            return

        corpus = \
            [record[Constants.CORPUS_FIELD] for record in records]
        dictionary = corpora.Dictionary.load(Constants.DICTIONARY_FILE)
        topic_model = ldamodel.LdaModel(
            corpus, id2word=dictionary,
            num_topics=Constants.TOPIC_MODEL_NUM_TOPICS,
            passes=Constants.TOPIC_MODEL_PASSES,
            iterations=Constants.TOPIC_MODEL_ITERATIONS)

        with open(topic_model_file_path, 'wb') as write_file:
            pickle.dump(topic_model, write_file, pickle.HIGHEST_PROTOCOL)

    elif Constants.TOPIC_MODEL_TYPE == 'ensemble':
        file_path = Constants.ENSEMBLED_RESULTS_FOLDER + \
                    "factors_final_k%02d.pkl" % Constants.TOPIC_MODEL_NUM_TOPICS

        if os.path.exists(file_path):
            print('Ensemble topic model already exists')
            return

        export_to_text(records)
        topic_ensemble_caller.run_local_parse_directory()
        topic_ensemble_caller.run_generate_kfold()
        topic_ensemble_caller.run_combine_nmf()

    else:
        raise ValueError('Unrecognized topic modeling algorithm: \'%s\'' %
                         Constants.TOPIC_MODEL_TYPE)


def train_context_extractor(records, stable=True):
    print('%s: train context topics model' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    if Constants.TOPIC_MODEL_TYPE == 'lda':
        context_extractor = LdaBasedContext(records)
        context_extractor.generate_review_corpus()
        context_extractor.build_topic_model()
        context_extractor.update_reviews_with_topics()
        context_extractor.get_context_rich_topics()
        context_extractor.clear_reviews()
    elif Constants.TOPIC_MODEL_TYPE == 'nmf':
        context_extractor = NmfContextExtractor(records)
        context_extractor.generate_review_bows()
        context_extractor.build_document_term_matrix()
        if stable:
            context_extractor.build_stable_topic_model()
        else:
            context_extractor.build_topic_model()
        context_extractor.update_reviews_with_topics()
        context_extractor.get_context_rich_topics()
        context_extractor.clear_reviews()
    else:
        raise ValueError('Unrecognized topic model type: \'%s\'' %
                         Constants.TOPIC_MODEL_TYPE)

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
    return create_topic_model(
        train_records, cycle_index, fold_index, check_exists)


def generate_file_with_commands():
    print('%s: Generating file with commands' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))

    code_path = constants.PYTHON_CODE_FOLDER[:-1]
    python_path = "PYTHONPATH='" + code_path + "' "
    python_command = "stdbuf -oL nohup python "
    base_file_name = "topicmodel-" + Constants.ITEM_TYPE
    log_file = " > ~/logs/" + base_file_name + "-%d-%d.log"
    commands_dir = expanduser("~") + "/tmp/"
    commands_file = commands_dir + base_file_name + ".sh"
    command_list = []

    for cycle in range(Constants.NUM_CYCLES):
        for fold in range(Constants.CROSS_VALIDATION_NUM_FOLDS):
            python_file = constants.PYTHON_CODE_FOLDER +\
                "topicmodeling/context/topic_model_creator.py -c %d -f %d"
            full_command = python_path + python_command + python_file + log_file
            command_list.append(full_command % (cycle, fold, cycle+1, fold+1))

    with open(commands_file, "w") as write_file:
        for command in command_list:
            write_file.write("%s\n" % command)


def export_to_text(records):
    print('%s: Exporting bag-of-words to text files' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))

    if not os.path.isdir(Constants.TEXT_FILES_FOLDER):
        os.mkdir(Constants.TEXT_FILES_FOLDER)

    folder = Constants.GENERATED_TEXT_FILES_FOLDER
    if not os.path.isdir(folder):
        os.mkdir(folder)

    topic_model_target = Constants.TOPIC_MODEL_TARGET_REVIEWS
    all_files_existed = True

    for record in records:

        if record[Constants.TOPIC_MODEL_TARGET_FIELD] != \
                topic_model_target and topic_model_target is not None:
            continue
        file_name = \
            folder + 'bow_' + str(record[Constants.REVIEW_ID_FIELD]) + '.txt'

        if os.path.exists(file_name):
            continue
        all_files_existed = False

        with codecs.open(file_name, 'w', encoding="utf-8-sig") as text_file:
            text_file.write(" ".join(record[Constants.BOW_FIELD]))

    if all_files_existed:
        print('Text files already existed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--cycle', metavar='int', type=int,
        nargs=1, help='The index of the running cycle')
    parser.add_argument(
        '-f', '--fold', metavar='int', type=int,
        nargs=1, help='The index of the cross validation fold')
    parser.add_argument(
        '-t', '--numtopics', metavar='int', type=int,
        nargs=1, help='The number of topics of the topic model')

    args = parser.parse_args()
    fold = args.fold[0] if args.fold is not None else None
    cycle = args.cycle[0] if args.cycle is not None else None
    num_topics = args.numtopics[0] if args.numtopics is not None else None

    if num_topics is not None:
        Constants.update_properties(
            {Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: num_topics})

    if fold is None and cycle is None:
        records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)

        if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
            num_records = len(records)
            records = records[:num_records / 2]
        print('num_reviews', len(records))

        create_topic_model(records, None, None)
    else:
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
