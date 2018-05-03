
import argparse
import time

import subprocess

from etl.reviews_preprocessor import ReviewsPreprocessor
from external.libfm import libfm_caller
from utils.constants import Constants, JAVA_CODE_FOLDER, PROPERTIES_FILE

JAVA_COMMAND = 'java'


def run_rival(task, dataset=None):
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
        Constants.RESULTS_FOLDER
    ]

    if dataset is not None:
        command.extend(['-s', dataset])

    print(jar_folder)
    print(command)

    p = subprocess.Popen(command, cwd=jar_folder)
    p.wait()


def full_cycle(evaluation_set):
    reviews_preprocessor = ReviewsPreprocessor(use_cache=True)
    reviews_preprocessor.full_cycle()
    run_rival('prepare_libfm')
    libfm_caller.main()
    run_rival('process_libfm_results', evaluation_set)


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
    args = parser.parse_args()
    num_topics = args.numtopics[0] if args.numtopics is not None else None
    item_type = args.itemtype[0] if args.itemtype is not None else None
    evaluation_set =\
        args.evaluationset[0] if args.evaluationset is not None else None

    if num_topics is not None:
        Constants.update_properties(
            {Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: num_topics})
    if item_type is not None:
        Constants.update_properties(
            {Constants.BUSINESS_TYPE_FIELD: item_type})

    full_cycle(evaluation_set)


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
