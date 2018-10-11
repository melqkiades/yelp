
import argparse
import time

import subprocess
import uuid

from os.path import expanduser

from etl.reviews_preprocessor import ReviewsPreprocessor
from external.carskit import carskit_caller
from external.libfm import libfm_caller
from utils import constants
from utils.constants import Constants, JAVA_CODE_FOLDER, PROPERTIES_FILE

JAVA_COMMAND = 'java'


def run_rival(task, evaluation_set=None, carskit_model=None):
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
        '-s',
        Constants.RIVAL_EVALUATION_STRATEGY,
        '-cf',
        Constants.CONTEXT_FORMAT
    ]

    if evaluation_set is not None:
        command.extend(['-e', evaluation_set])

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
        'process_carskit_results', evaluation_set,
        carskit_model=Constants.CARSKIT_RECOMMENDERS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-k', '--numtopics', metavar='int', type=int,
        nargs=1, help='The number of topics of the topic model')
    parser.add_argument(
        '-i', '--itemtype', metavar='string', type=str,
        nargs=1, help='The type of items')
    parser.add_argument(
        '-s', '--strategy', metavar='string', type=str,
        nargs=1, help='The evaluation strategy (user_test or rel_plus_n)')
    parser.add_argument(
        '-e', '--evaluationset', metavar='string', type=str,
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
    strategy =\
        args.strategy[0] if args.strategy is not None else None
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
    if strategy is not None:
        Constants.update_properties(
            {Constants.RIVAL_EVALUATION_STRATEGY_FIELD: strategy})
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


def generate_execution_scripts():
    print('%s: Generating recommender_runner scripts' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))

    algorithms = [
        'carskit_CAMF_C',
        'carskit_CAMF_CI',
        'carskit_CAMF_CU',
        'carskit_CAMF_CUCI',
        'carskit_DCR',
        'carskit_DCW',
        'libfm',
    ]

    algorithm_params_map = {
        'carskit_CAMF_C': {'num.factors': [10, 20, 30, 40]},
        'carskit_CAMF_CI': {'num.factors': [10, 20, 30, 40]},
        'carskit_CAMF_CU': {'num.factors': [10, 20, 30, 40]},
        'carskit_CAMF_CUCI': {'num.factors': [10, 20, 30, 40]},
        'carskit_DCR': {
            'DCR': [
                '-wt 0.9 -wd 0.4 -p 3 -lp 2.05 -lg 2.05',
                '-wt 0.9 -wd 0.4 -p 5 -lp 2.05 -lg 2.05',
            ]
        },
        'carskit_DCW': {
            'DCW': [
                '-wt 0.9 -wd 0.4 -p 3 -lp 2.05 -lg 2.05 -th 0.5',
                '-wt 0.9 -wd 0.4 -p 3 -lp 2.05 -lg 2.05 -th 0.8',
                '-wt 0.9 -wd 0.4 -p 5 -lp 2.05 -lg 2.05 -th 0.5',
                '-wt 0.9 -wd 0.4 -p 5 -lp 2.05 -lg 2.05 -th 0.8',
            ]
        },
        'libfm': {}
    }

    strategies = ['user_test', 'rel_plus_n']
    evaluation_sets = ['test_users', 'test_only_users']
    context_formats = ['predefined_context', 'top_words']

    code_path = constants.PYTHON_CODE_FOLDER
    python_path = "PYTHONPATH='" + code_path[:-1] + "' "
    python_command = "stdbuf -oL nohup python "
    base_file_name = "recommender_runner-" + Constants.ITEM_TYPE
    log_file = " > ~/logs/" + base_file_name + "_%d_%s.log"
    commands_dir = expanduser("~") + "/tmp/"
    commands_file = commands_dir + base_file_name + ".sh"
    command_list = []
    i = 1

    for strategy in strategies:
        for evaluation_set in evaluation_sets:
            for context_format in context_formats:
                for algorithm in algorithms:
                    for param_name, param_values in algorithm_params_map[algorithm].items():
                        for param_value in param_values:
                            # print(algorithm, param_name, param_value)

                            params = "'%s=%s'" % (param_name, param_value)

                            python_file = code_path + \
                                          "main/recommender_runner.py -k %s -i %s -s %s -e %s -cf %s -a %s -cp %s" % (
                                    Constants.TOPIC_MODEL_NUM_TOPICS,
                                    Constants.ITEM_TYPE,
                                    strategy,
                                    evaluation_set,
                                    context_format,
                                    algorithm,
                                    params,
                            )

                            full_command =\
                                python_path + python_command + python_file +\
                                log_file % (i, uuid.uuid4().hex)
                            # print(python_file)
                            # print(full_command)
                            command_list.append(full_command)
                            i += 1
    #
    with open(commands_file, "w") as write_file:
        for command in command_list:
            write_file.write("%s\n" % command)


# start = time.time()
# main()
# generate_execution_scripts()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print("Total time = %f seconds" % total_time)
