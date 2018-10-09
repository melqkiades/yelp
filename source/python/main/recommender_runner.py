
import argparse
import time

import subprocess

from etl.reviews_preprocessor import ReviewsPreprocessor
from external.carskit import carskit_caller
from external.libfm import libfm_caller
from utils.constants import Constants, JAVA_CODE_FOLDER, PROPERTIES_FILE

JAVA_COMMAND = 'java'


def run_rival(task, dataset=None, carskit_model=None):
    print('%s: Run RiVaL' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    jar_file = 'richcontext-1.0-SNAPSHOT-jar-with-dependencies.jar'
    jar_folder = JAVA_CODE_FOLDER + 'richcontext/target/'

    command = [
        JAVA_COMMAND,
        '-jar',
        jar_file,
        '-t',
        task,
        '-i',
        Constants.ITEM_TYPE,
        '-k',
        str(Constants.TOPIC_MODEL_NUM_TOPICS),
        '-p',
        PROPERTIES_FILE,
        '-d',
        Constants.CACHE_FOLDER,
        '-o',
        Constants.RESULTS_FOLDER,
        '-cf',
        Constants.CONTEXT_FORMAT
    ]

    if dataset is not None:
        command.extend(['-s', dataset])

    if carskit_model is not None:
        command.extend(['-carskit_model', carskit_model])

    if Constants.CARSKIT_PARAMETERS is not None:
        command.extend(['-carskit_params', Constants.CARSKIT_PARAMETERS])

    print(jar_folder)
    print(" ".join(command))

    p = subprocess.Popen(command, cwd=jar_folder)
    p.wait()


def full_cycle_libfm(evaluation_set):
    reviews_preprocessor = ReviewsPreprocessor(use_cache=True)
    reviews_preprocessor.full_cycle()
    run_rival('prepare_libfm')
    libfm_caller.main()
    run_rival('process_libfm_results', evaluation_set)


def full_cycle_carskit(evaluation_set):
    reviews_preprocessor = ReviewsPreprocessor(use_cache=True)
    reviews_preprocessor.full_cycle()
    run_rival('prepare_carskit')
    carskit_caller.main()
    run_rival(
        'process_carskit_results', carskit_model=Constants.CARSKIT_RECOMMENDERS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-k', '--numtopics', metavar='int', type=int,
        nargs=1, help='The number of topics of the topic model')
    parser.add_argument(
        '-i', '--itemtype', metavar='string', type=str,
        nargs=1, help='The type of items')
    parser.add_argument(
        '-s', '--evaluationset', metavar='string', type=str,
        nargs=1, help='The evaluation set')
    parser.add_argument(
        '-cf', '--contextformat', metavar='string', type=str, nargs=1,
        help='The strategy to extract the contextual information')
    parser.add_argument(
        '-a', '--algorithm', metavar='string', type=str,
        nargs=1, help='The algorithm used to produce recommendations')
    parser.add_argument(
        '-cp', '--carskitparams', metavar='string', type=str,
        nargs=1, help='The hyperparameters for the CARSKit model')
    args = parser.parse_args()
    num_topics = args.numtopics[0] if args.numtopics is not None else None
    item_type = args.itemtype[0] if args.itemtype is not None else None
    evaluation_set =\
        args.evaluationset[0] if args.evaluationset is not None else None
    context_format =\
        args.contextformat[0] if args.contextformat is not None else None
    algorithm =\
        args.algorithm[0] if args.algorithm is not None else 'libfm'
    carskit_params =\
        args.carskitparams[0] if args.carskitparams is not None else None

    if num_topics is not None:
        Constants.update_properties(
            {Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: num_topics})
    if item_type is not None:
        Constants.update_properties(
            {Constants.BUSINESS_TYPE_FIELD: item_type})
    if context_format is not None:
        Constants.update_properties(
            {Constants.CONTEXT_FORMAT_FIELD: context_format})
    if carskit_params is not None:
        Constants.update_properties(
            {Constants.CARSKIT_PARAMETERS_FIELD: carskit_params})
    if algorithm.startswith('carskit_'):
        carskit_recommender = algorithm.split('carskit_')[1]
        Constants.update_properties(
            {'carskit_recommenders': carskit_recommender})
        full_cycle_carskit(evaluation_set)
    elif algorithm == 'libfm':
        full_cycle_libfm(evaluation_set)
    else:
        raise ValueError('Unknown algorithm \'%s\'' % algorithm)



# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print("Total time = %f seconds" % total_time)
