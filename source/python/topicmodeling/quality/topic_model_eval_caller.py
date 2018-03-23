import csv
import glob
import uuid

import time

import subprocess

from topicmodeling.context import topic_model_analyzer
from utils.constants import Constants


PYTHON_COMMAND = 'python'
BASE_FOLDER = Constants.TOPIC_MODEL_FOLDER + 'base/'
OUTPUT_FOLDER = Constants.GENERATED_FOLDER
TERM_STABILITY = 'term_stability'
TERM_DIFFERENCE = 'term_difference'
METRIC_MAP = {
    TERM_STABILITY: 'stability',
    TERM_DIFFERENCE: 'diff',
}


def run_eval_topic_model(metric):

    parse_directory_command = Constants.TOPIC_ENSEMBLE_FOLDER + \
        'eval-' + metric.replace('_', '-') + '.py'

    csv_file = Constants.generate_file_name(
        metric, 'csv', BASE_FOLDER, None, None, True, True)

    dataset_file_name = Constants.generate_file_name(
        'topic_model', '', BASE_FOLDER, None, None, True, True)[:-1] +\
        '/ranks*.pkl'
    topic_model_files = glob.glob(dataset_file_name)

    command = [
        PYTHON_COMMAND,
        parse_directory_command,
    ]
    command.extend(topic_model_files)
    command.extend([
        '-o',
        csv_file
    ])

    print(' '.join(command))

    unique_id = uuid.uuid4().hex
    log_file_name = Constants.GENERATED_FOLDER + Constants.ITEM_TYPE + '_' + \
        Constants.TOPIC_MODEL_TARGET_REVIEWS + '_' + metric + '_' +\
        unique_id + '.log'
    #
    log_file = open(log_file_name, "w")
    p = subprocess.Popen(
        command, stdout=log_file, cwd=Constants.TOPIC_ENSEMBLE_FOLDER)
    p.wait()

    results = read_csv_first_column_as_key(csv_file, metric)
    results[Constants.TOPIC_MODEL_NUM_TOPICS_FIELD] =\
        Constants.TOPIC_MODEL_NUM_TOPICS
    results[Constants.TOPIC_MODEL_TYPE_FIELD] = Constants.TOPIC_MODEL_TYPE

    return results


def read_csv_first_column_as_key(csv_file, metric):
    result = {}
    with open(csv_file, 'r') as f:
        dictionary = csv.DictReader(f)
        for row in dictionary:
            metric_key = METRIC_MAP[metric]
            result.setdefault(
                metric + row['statistic'], []).append(row[metric_key])

    return result


def cycle_eval_topic_model(metric, num_topics_list):

    csv_file_name = Constants.generate_file_name(
        metric, 'csv', Constants.RESULTS_FOLDER, None, None,
        False)

    for topic in num_topics_list:
        Constants.update_properties(
            {Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: topic})
        results = run_eval_topic_model(metric)
        topic_model_analyzer.write_results_to_csv(csv_file_name, results)


def main():
    # print(run_eval_topic_model())
    # print(cycle_eval_topic_model(TERM_STABILITY))
    topics = range(2, 44)
    cycle_eval_topic_model(TERM_DIFFERENCE, topics)

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
