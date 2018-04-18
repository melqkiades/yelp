
# Call parse directory
import glob
import os
import subprocess

import time
import uuid

import shutil

from utils.constants import Constants, PYTHON_CODE_FOLDER

PYTHON_COMMAND = 'python'
BASE_FOLDER = Constants.TOPIC_MODEL_FOLDER + 'base/'
CORPUS_FOLDER = Constants.TOPIC_MODEL_FOLDER + 'corpus/'


# TODO: Consider moving this to the Constants class
DATASET_FILE_NAME = Constants.generate_file_name(
    'topic_ensemble_corpus', '', CORPUS_FOLDER, None, None, False)[:-1]
# def get_dataset_file_name():
#     return Constants.CACHE_FOLDER + Constants.ITEM_TYPE + '_' + \
#         Constants.TOPIC_MODEL_TARGET_REVIEWS + '_document_term_matrix'


# TODO: Consider moving this to the Constants class
def get_topic_model_prefix(folder='', seed=None):

    prefix = 'topic_model'
    if seed is not None:
        prefix += '_seed-' + str(seed)

    return Constants.generate_file_name(
        prefix, '', folder, None, None, True, True)[:-1]


def run_parse_directory():

    parse_directory_command = Constants.TOPIC_ENSEMBLE_FOLDER + \
        'parse-directory.py'

    command = [
        PYTHON_COMMAND,
        parse_directory_command,
        Constants.GENERATED_TEXT_FILES_FOLDER,
        '-o',
        DATASET_FILE_NAME,
        '--tfidf',
        '--norm',
    ]

    print(command)

    unique_id = uuid.uuid4().hex
    log_file_name = Constants.GENERATED_FOLDER + Constants.ITEM_TYPE + '_' + \
        Constants.TOPIC_MODEL_TARGET_REVIEWS + '_parse_directory_' +\
        unique_id + '.log'

    log_file = open(log_file_name, "w")
    p = subprocess.Popen(
        command, stdout=log_file, cwd=Constants.TOPIC_ENSEMBLE_FOLDER)
    p.wait()


def run_local_parse_directory():

    parse_directory_command = PYTHON_CODE_FOLDER + 'topicmodeling/' \
        'belford_tfidf.py'

    if not os.path.isdir(Constants.TOPIC_MODEL_FOLDER):
        os.mkdir(Constants.TOPIC_MODEL_FOLDER)

    if not os.path.isdir(BASE_FOLDER):
        os.mkdir(BASE_FOLDER)

    if not os.path.isdir(CORPUS_FOLDER):
        os.mkdir(CORPUS_FOLDER)

    command = [
        PYTHON_COMMAND,
        parse_directory_command,
        Constants.GENERATED_TEXT_FILES_FOLDER,
        '-o',
        DATASET_FILE_NAME,
        '--tfidf',
        '--norm',
        '--df',
        str(Constants.MIN_DICTIONARY_WORD_COUNT),
        '--minlen',
        '10'
    ]

    print(command)

    unique_id = uuid.uuid4().hex
    log_file_name = Constants.GENERATED_FOLDER + Constants.ITEM_TYPE + '_' + \
        str(Constants.TOPIC_MODEL_TARGET_REVIEWS) + '_parse_directory_' +\
        unique_id + '.log'

    log_file = open(log_file_name, "w")
    p = subprocess.Popen(
        command, stdout=log_file, cwd=Constants.TOPIC_ENSEMBLE_FOLDER)
    p.wait()


# Call generate nmf or generate kfold
def run_generate_nmf():
    generate_nmf_command = Constants.TOPIC_ENSEMBLE_FOLDER + \
        'generate-nmf.py'
    output_folder = BASE_FOLDER + get_topic_model_prefix() + '/'

    if not os.path.isdir(Constants.TOPIC_MODEL_FOLDER):
        os.mkdir(Constants.TOPIC_MODEL_FOLDER)

    if not os.path.isdir(BASE_FOLDER):
        os.mkdir(BASE_FOLDER)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    command = [
        PYTHON_COMMAND,
        generate_nmf_command,
        DATASET_FILE_NAME + '.pkl',
        '-k',
        str(Constants.TOPIC_MODEL_NUM_TOPICS),
        '-r',
        str(Constants.TOPIC_MODEL_PASSES),
        '-o',
        output_folder,
    ]

    print(command)

    unique_id = uuid.uuid4().hex
    log_file_name = Constants.GENERATED_FOLDER + get_topic_model_prefix() + \
        '_parse_directory_' + unique_id + '.log'

    log_file = open(log_file_name, "w")
    p = subprocess.Popen(
        command, stdout=log_file, cwd=Constants.TOPIC_ENSEMBLE_FOLDER)
    p.wait()


# Call generate kfold
def run_generate_kfold(seed=None):
    generate_nmf_command = Constants.TOPIC_ENSEMBLE_FOLDER + \
        'generate-kfold.py'
    base_topic_folder = get_topic_model_prefix(seed=seed)
    output_folder = BASE_FOLDER + base_topic_folder + '/'

    if not os.path.isdir(Constants.TOPIC_MODEL_FOLDER):
        os.mkdir(Constants.TOPIC_MODEL_FOLDER)

    if not os.path.isdir(BASE_FOLDER):
        os.mkdir(BASE_FOLDER)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    command = [
        PYTHON_COMMAND,
        generate_nmf_command,
        DATASET_FILE_NAME + '.pkl',
        '-k',
        str(Constants.TOPIC_MODEL_NUM_TOPICS),
        '-r',
        str(Constants.TOPIC_MODEL_PASSES),
        '-f',
        str(Constants.TOPIC_MODEL_FOLDS),
        '-o',
        output_folder,
    ]

    if seed is not None:
        command.extend([
            '--seed',
            str(seed)
        ])

    print(command)

    unique_id = uuid.uuid4().hex
    log_file_name = Constants.GENERATED_FOLDER + base_topic_folder + \
        '_parse_directory_' + unique_id + '.log'

    log_file = open(log_file_name, "w")
    p = subprocess.Popen(
        command, stdout=log_file, cwd=Constants.TOPIC_ENSEMBLE_FOLDER)
    p.wait()


# Call combine nmf
def run_combine_nmf(seed=None):
    generate_nmf_command = Constants.TOPIC_ENSEMBLE_FOLDER + \
        'combine-nmf.py'
    base_topic_folder = get_topic_model_prefix(seed=seed)
    base_files = \
        glob.glob(BASE_FOLDER + base_topic_folder + '/*factors*.pkl')
    output_folder = \
        get_topic_model_prefix(Constants.ENSEMBLE_FOLDER, seed) + '/'

    if not os.path.isdir(Constants.ENSEMBLE_FOLDER):
        os.mkdir(Constants.ENSEMBLE_FOLDER)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    command = [
        PYTHON_COMMAND,
        generate_nmf_command,
        DATASET_FILE_NAME + '.pkl',
        ]
    command.extend(base_files)
    command.extend([
        '-k',
        str(Constants.TOPIC_MODEL_NUM_TOPICS),
        '-o',
        output_folder,
    ])

    if seed is not None:
        command.extend([
            '--seed',
            str(seed)
        ])

    print(command)

    unique_id = uuid.uuid4().hex
    log_file_name = Constants.GENERATED_FOLDER + base_topic_folder + \
        '_parse_directory_' + unique_id + '.log'

    log_file = open(log_file_name, "w")
    p = subprocess.Popen(
        command, stdout=log_file, cwd=Constants.TOPIC_ENSEMBLE_FOLDER)
    p.wait()

    shutil.rmtree(BASE_FOLDER + base_topic_folder)


def create_several_topic_models():

    run_local_parse_directory()

    random_seeds = range(1, 101)
    total_topic_models = len(random_seeds)
    for seed in random_seeds:
        print('\n\n\nCreating %d of %d topic models' % (seed, total_topic_models))
        run_generate_kfold(seed)
        run_combine_nmf(seed)


def main():

    # run_parse_directory()
    run_local_parse_directory()
    # run_generate_nmf()
    run_generate_kfold()
    run_combine_nmf()

    # create_several_topic_models()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
