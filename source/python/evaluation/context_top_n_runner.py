import copy
import csv
import json
import os
import random
from subprocess import call
import time
import cPickle as pickle
import uuid
import gc
import numpy
# from fastFM import als
# from fastFM import mcmc
# from fastFM import sgd
from gensim import corpora

from etl import ETLUtils
from etl import libfm_converter
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
# from recommenders import fastfm_recommender
from topicmodeling.context import topic_model_analyzer
from topicmodeling.context import topic_model_creator
from topicmodeling.context.lda_based_context import LdaBasedContext
from tripadvisor.fourcity import extractor
from utils.constants import Constants

__author__ = 'fpena'


basic_headers = [
    Constants.RATING_FIELD,
    Constants.USER_ID_FIELD,
    Constants.ITEM_ID_FIELD
]


def build_headers(context_rich_topics):
    headers = basic_headers[:]
    for topic in context_rich_topics:
        topic_id = 'topic' + str(topic[0])
        headers.append(topic_id)
    if Constants.USE_NO_CONTEXT_TOPICS_SUM:
        headers.append('nocontexttopics')
    return headers


def create_user_item_map(records):
    user_ids = extractor.get_groupby_list(records, Constants.USER_ID_FIELD)
    user_item_map = {}
    user_count = 0

    for user_id in user_ids:
        user_records =\
            ETLUtils.filter_records(records, Constants.USER_ID_FIELD, [user_id])
        user_items =\
            extractor.get_groupby_list(user_records, Constants.ITEM_ID_FIELD)
        user_item_map[user_id] = user_items
        user_count += 1

        # print("user count %d" % user_count),
        print 'user count: {0}\r'.format(user_count),

    print

    return user_item_map


def run_libfm(train_file, test_file, predictions_file, log_file):

    libfm_command = Constants.LIBFM_FOLDER + 'libFM'

    command = [
        libfm_command,
        '-task',
        'r',
        '-train',
        train_file,
        '-test',
        test_file,
        '-dim',
        ','.join(map(str,
                     [Constants.FM_USE_BIAS, Constants.FM_USE_1WAY_INTERACTIONS,
                      Constants.FM_NUM_FACTORS])),
        '-init_stdev',
        str(Constants.FM_INIT_STDEV),
        '-iter',
        str(Constants.FM_ITERATIONS),
        '-out',
        predictions_file
    ]

    print(command)

    if Constants.LIBFM_SEED is not None:
        command.extend(['-seed', str(Constants.LIBFM_SEED)])

    f = open(log_file, "w")
    call(command, stdout=f)


def filter_reviews(records, reviews, review_type):
    print('filter: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    if not review_type:
        return records, reviews

    filtered_records = []
    filtered_reviews = []

    for record, review in zip(records, reviews):
        if record[Constants.PREDICTED_CLASS_FIELD] == review_type:
            filtered_records.append(record)
            filtered_reviews.append(review)

    return filtered_records, filtered_reviews


def write_results_to_csv(results):
    if not os.path.exists(Constants.CSV_RESULTS_FILE):
        with open(Constants.CSV_RESULTS_FILE, 'w') as f:
            w = csv.DictWriter(f, results.keys())
            w.writeheader()
            w.writerow(results)
    else:
        with open(Constants.CSV_RESULTS_FILE, 'a') as f:
            w = csv.DictWriter(f, results.keys())
            w.writerow(results)


def write_results_to_json(results):
    if not os.path.exists(Constants.JSON_RESULTS_FILE):
        with open(Constants.JSON_RESULTS_FILE, 'w') as f:
            json.dump(results, f)
            f.write('\n')
    else:
        with open(Constants.JSON_RESULTS_FILE, 'a') as f:
            json.dump(results, f)
            f.write('\n')


class ContextTopNRunner(object):

    def __init__(self):
        self.records = None
        self.original_records = None
        self.train_records = None
        self.test_records = None
        self.records_to_predict = None
        self.predictions = None
        self.top_n_evaluator = None
        self.headers = None
        self.important_records = None
        self.context_rich_topics = []
        self.context_topics_map = None
        self.csv_train_file = None
        self.csv_test_file = None
        self.context_predictions_file = None
        self.context_train_file = None
        self.context_test_file = None
        self.context_log_file = None

    def clear(self):
        print('clear: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        # self.records = None
        self.train_records = None
        self.test_records = None
        self.records_to_predict = None
        self.top_n_evaluator = None
        self.headers = None
        self.important_records = None
        self.context_rich_topics = []
        self.context_topics_map = None

        if Constants.SOLVER == Constants.LIBFM:
            os.remove(self.csv_train_file)
            os.remove(self.csv_test_file)
            os.remove(self.context_predictions_file)
            os.remove(self.context_train_file)
            os.remove(self.context_test_file)
            os.remove(self.context_log_file)

        self.csv_train_file = None
        self.csv_test_file = None
        self.context_predictions_file = None
        self.context_train_file = None
        self.context_test_file = None
        self.context_log_file = None
        gc.collect()

    def create_tmp_file_names(self):

        unique_id = uuid.uuid4().hex
        prefix = Constants.GENERATED_FOLDER + unique_id + '_' + \
            Constants.ITEM_TYPE
        # prefix = constants.GENERATED_FOLDER + constants.ITEM_TYPE

        print('unique id: %s' % unique_id)

        self.csv_train_file = prefix + '_train.csv'
        self.csv_test_file = prefix + '_test.csv'
        self.context_predictions_file = prefix + '_predictions.txt'
        self.context_train_file = self.csv_train_file + '.libfm'
        self.context_test_file = self.csv_test_file + '.libfm'
        self.context_log_file = prefix + '.log'

    @staticmethod
    def plant_seeds():

        if Constants.RANDOM_SEED is not None:
            print('random seed: %d' % Constants.RANDOM_SEED)
            random.seed(Constants.RANDOM_SEED)
        if Constants.NUMPY_RANDOM_SEED is not None:
            print('numpy random seed: %d' % Constants.NUMPY_RANDOM_SEED)
            numpy.random.seed(Constants.NUMPY_RANDOM_SEED)

    def load(self):
        print('load: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        self.original_records =\
            ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)
        # ETLUtils.drop_fields(['tagged_words'], self.original_records)
        print('num_records: %d' % len(self.original_records))

        if not os.path.exists(Constants.USER_ITEM_MAP_FILE):
            records = ETLUtils.load_json_file(Constants.RECORDS_FILE)
            user_item_map = create_user_item_map(records)
            with open(Constants.USER_ITEM_MAP_FILE, 'wb') as write_file:
                pickle.dump(user_item_map, write_file, pickle.HIGHEST_PROTOCOL)

    def shuffle(self, records):
        print('shuffle: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        random.shuffle(records)

    def get_records_to_predict_topn(self):
        print(
            'get_records_to_predict_topn: %s'
            % time.strftime("%Y/%m/%d-%H:%M:%S")
        )

        with open(Constants.USER_ITEM_MAP_FILE, 'rb') as read_file:
            user_item_map = pickle.load(read_file)

        self.top_n_evaluator = TopNEvaluator(
            self.records, self.test_records, Constants.ITEM_TYPE, 10,
            Constants.TOPN_NUM_ITEMS)
        self.top_n_evaluator.initialize(user_item_map)
        self.important_records = self.top_n_evaluator.important_records

        if Constants.TEST_CONTEXT_REVIEWS_ONLY:
            self.important_records = self.filter_context_words(
                self.important_records)

            self.records_to_predict =\
                self.top_n_evaluator.get_records_to_predict()

        if Constants.MAX_SAMPLE_TEST_SET is not None:
            print('important_records %d' % len(self.important_records))
            if len(self.important_records) > Constants.MAX_SAMPLE_TEST_SET:
                self.important_records = random.sample(
                    self.important_records, Constants.MAX_SAMPLE_TEST_SET)
            else:
                message = 'WARNING max_sample_test_set is greater than the ' \
                          'number of important records'
                print(message)

        self.top_n_evaluator.important_records = self.important_records
        self.records_to_predict = self.top_n_evaluator.get_records_to_predict()
        self.test_records = None
        gc.collect()

    def get_records_to_predict_rmse(self):
        print(
            'get_records_to_predict_rmse: %s' %
            time.strftime("%Y/%m/%d-%H:%M:%S")
        )
        self.important_records = self.test_records

        if Constants.TEST_CONTEXT_REVIEWS_ONLY:
            self.important_records = self.filter_context_words(
                self.test_records)

        self.records_to_predict = self.important_records
        self.test_records = None
        gc.collect()

    def get_records_to_predict(self):

        if Constants.EVALUATION_METRIC == 'topn_recall':
            self.get_records_to_predict_topn()
        elif Constants.EVALUATION_METRIC == 'rmse':
            self.get_records_to_predict_rmse()
        else:
            raise ValueError('Unrecognized evaluation metric')

    def train_topic_model(self, cycle_index, fold_index):

        if Constants.CACHE_TOPIC_MODEL:
            print('loading topic model')
            lda_based_context = topic_model_creator.load_topic_model(
                cycle_index, fold_index)
        else:
            print('train topic model: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

            lda_based_context = LdaBasedContext(self.train_records)
            lda_based_context.generate_review_corpus()
            lda_based_context.build_topic_model()
            lda_based_context.update_reviews_with_topics()
            lda_based_context.get_context_rich_topics()

        self.context_rich_topics = lda_based_context.context_rich_topics
        print('Trained LDA Model: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        return lda_based_context

    def find_reviews_topics(self, lda_based_context):
        print('find topics: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        lda_based_context.find_contextual_topics(self.train_records)

        lda_based_context.find_contextual_topics(
            self.important_records, Constants.TEXT_SAMPLING_PROPORTION)

        self.context_topics_map = {}
        for record in self.important_records:
            self.context_topics_map[record[Constants.REVIEW_ID_FIELD]] = \
                record[Constants.CONTEXT_TOPICS_FIELD]

        # self.train_records = self.filter_context_words(self.train_records)
        # self.print_context_topics(self.important_records)

        self.important_records = None
        gc.collect()

    @staticmethod
    def print_context_topics(records):
        dictionary = corpora.Dictionary.load(Constants.DICTIONARY_FILE)

        all_context_topics = set()

        for record in records:
            words = []
            corpus = record[Constants.CORPUS_FIELD]

            for element in corpus:
                word_id = element[0]
                word = dictionary[word_id]
                words.append(word + ' (' + str(word_id) + ')')

            context_topics = record[Constants.CONTEXT_TOPICS_FIELD]
            used_context_topics =\
                dict((k, v) for k, v in context_topics.items() if v > 0.0)
            all_context_topics |= set(used_context_topics.keys())

            print('words: %s' % ', '.join(words))
            print('text: %s' % record[Constants.TEXT_FIELD])
            print('bow', record[Constants.BOW_FIELD])
            # print('pos tags', record[Constants.POS_TAGS_FIELD])
            print(record[Constants.TOPICS_FIELD])
            # # print(record[Constants.CONTEXT_TOPICS_FIELD])
            print(used_context_topics)
            # print('')

        # print('important records: %d' % len(records))
        # print('context records: %d' % len(context_records))
        # print('no context records: %d' % len(no_context_records))
        # print('all used context words', all_context_words)
        print('all used context topics', all_context_topics)
        # print('all used context words count: %d' % len(all_context_words))
        print('all used context topics: %d' % len(all_context_topics))

    @staticmethod
    def filter_context_words(records):
        """
        Takes a list of records and based on the 'corpus' field looks at which
        records have contextual words (the ones that appear in the manually
        defined set in topics_analyzer.all_context_words) and returns a list
        with those records

        :param records: a list of reviews
        :return: a list of the records that contain contextual words
        """
        dictionary = corpora.Dictionary.load(Constants.DICTIONARY_FILE)

        no_context_records = []
        context_records = []
        item_type = Constants.ITEM_TYPE
        # print(topic_model_analyzer.all_context_words[item_type])

        for record in records:
            words = []
            context_corpus = []
            context_words = []
            corpus = record[Constants.CORPUS_FIELD]

            for element in corpus:
                word_id = element[0]
                word = dictionary[word_id]
                words.append(word + ' (' + str(word_id) + ')')
                if word in topic_model_analyzer.all_context_words[item_type]:
                    context_corpus.append(element)
                    context_words.append(word + ' (' + str(word_id) + ')')
            record[Constants.CORPUS_FIELD] = context_corpus

            if len(context_corpus) == 0:
                no_context_records.append(record)
            else:
                context_records.append(record)

            # print('words: %s' % ', '.join(words))
            # print('context words: %s' % ', '.join(context_words))
            # print('context corpus', context_corpus)

        print('important records: %d' % len(records))
        print('context records: %d' % len(context_records))
        print('no context records: %d' % len(no_context_records))
        return context_records

    def prepare_records_for_libfm(self):
        print('prepare_records_for_libfm: %s' %
              time.strftime("%Y/%m/%d-%H:%M:%S"))

        self.headers = build_headers(self.context_rich_topics)

        if Constants.FM_REVIEW_TYPE == Constants.SPECIFIC or \
                Constants.FM_REVIEW_TYPE == Constants.GENERIC:
            self.train_records = ETLUtils.filter_records(
                self.train_records, Constants.PREDICTED_CLASS_FIELD,
                [Constants.FM_REVIEW_TYPE])

        with open(self.csv_train_file, 'w') as out_file:
            writer = csv.writer(out_file)

            # Write header
            writer.writerow(self.headers)

            for record in self.train_records:
                row = []
                for header in basic_headers:
                    row.append(record[header])

                if Constants.USE_CONTEXT is True:
                    context_topics = record[Constants.CONTEXT_TOPICS_FIELD]
                    for topic in self.context_rich_topics:
                        # print('context_topics', context_topics)
                        row.append(context_topics['topic' + str(topic[0])])

                    if Constants.USE_NO_CONTEXT_TOPICS_SUM:
                        row.append(context_topics['nocontexttopics'])

                writer.writerow(row)

        self.train_records = None
        gc.collect()

        with open(self.csv_test_file, 'w') as out_file:
            writer = csv.writer(out_file)

            # Write header
            writer.writerow(self.headers)

            for record in self.records_to_predict:
                row = []
                for header in basic_headers:
                    row.append(record[header])

                if Constants.USE_CONTEXT is True:
                    important_record = record[Constants.REVIEW_ID_FIELD]
                    context_topics = \
                        self.context_topics_map[important_record]
                    for topic in self.context_rich_topics:
                        row.append(context_topics['topic' + str(topic[0])])

                    if Constants.USE_NO_CONTEXT_TOPICS_SUM:
                        row.append(context_topics['nocontexttopics'])

                writer.writerow(row)

        # self.records_to_predict = None
        self.context_topics_map = None
        self.context_rich_topics = None
        gc.collect()

        print('Exported CSV and JSON files: %s'
              % time.strftime("%Y/%m/%d-%H:%M:%S"))

        csv_files = [
            self.csv_train_file,
            self.csv_test_file
        ]

        print('num_cols', len(self.headers))

        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], [], ',', has_header=True,
            suffix='.libfm')

        print('Exported LibFM files: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    # def predict_fastfm(self):
    #
    #     if Constants.USE_CONTEXT:
    #         for record in self.records_to_predict:
    #             important_record = record[Constants.REVIEW_ID_FIELD]
    #             record[Constants.CONTEXT_TOPICS_FIELD] = \
    #                 self.context_topics_map[important_record]
    #
    #     all_records = self.train_records + self.records_to_predict
    #     x_matrix, y_vector = fastfm_recommender.records_to_matrix(
    #         all_records, self.context_rich_topics)
    #
    #     encoder = OneHotEncoder(categorical_features=[0, 1], sparse=True)
    #     encoder.fit(x_matrix)
    #
    #     x_train = encoder.transform(x_matrix[:len(self.train_records)])
    #     y_train = y_vector[:len(self.train_records)]
    #     x_test = encoder.transform(x_matrix[len(self.train_records):])
    #
    #     if Constants.FASTFM_METHOD == 'mcmc':
    #         # solver = mcmc.FMRegression(n_iter=num_iters, rank=num_factors)
    #         solver = mcmc.FMRegression(rank=Constants.FM_NUM_FACTORS)
    #         self.predictions = solver.fit_predict(x_train, y_train, x_test)
    #     elif Constants.FASTFM_METHOD == 'als':
    #         solver = als.FMRegression(rank=Constants.FM_NUM_FACTORS)
    #         solver.fit(x_train, y_train)
    #         self.predictions = solver.predict(x_test)
    #     elif Constants.FASTFM_METHOD == 'sgd':
    #         solver = sgd.FMRegression(rank=Constants.FM_NUM_FACTORS)
    #         solver.fit(x_train, y_train)
    #         self.predictions = solver.predict(x_test)

    def predict_libfm(self):
        print('predict: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        run_libfm(
            self.context_train_file, self.context_test_file,
            self.context_predictions_file, self.context_log_file)
        self.predictions = rmse_calculator.read_targets_from_txt(
            self.context_predictions_file)

    def predict(self):
        if Constants.SOLVER == Constants.LIBFM:
            self.prepare_records_for_libfm()
            self.predict_libfm()
        # elif Constants.SOLVER == Constants.FASTFM:
        #     self.predict_fastfm()

    def evaluate_topn(self):
        print('evaluate_topn: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        self.top_n_evaluator.evaluate(self.predictions)
        recall = self.top_n_evaluator.recall

        print('Recall: %f' % recall)
        print('Specific recall: %f' % self.top_n_evaluator.specific_recall)
        print('Generic recall: %f' % self.top_n_evaluator.generic_recall)

        return recall, self.top_n_evaluator.specific_recall,\
            self.top_n_evaluator.generic_recall

    def evaluate_rmse(self):
        print('evaluate_topn: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))

        true_values = [
            record[Constants.RATING_FIELD] for record in self.records_to_predict
        ]

        specific_true_values = []
        specific_predictions = []
        generic_true_values = []
        generic_predictions = []

        index = 0
        for record, prediction in zip(
                self.records_to_predict, self.predictions):
            if record[Constants.PREDICTED_CLASS_FIELD] == 'specific':
                specific_true_values.append(record[Constants.RATING_FIELD])
                specific_predictions.append(prediction)
            elif record[Constants.PREDICTED_CLASS_FIELD] == 'generic':
                generic_true_values.append(record[Constants.RATING_FIELD])
                generic_predictions.append(prediction)
            index += 1

        rmse = rmse_calculator.calculate_rmse(true_values, self.predictions)
        specific_rmse = rmse_calculator.calculate_rmse(
            specific_true_values, specific_predictions)
        generic_rmse = rmse_calculator.calculate_rmse(
            generic_true_values, generic_predictions)

        print('RMSE: %f' % rmse)
        print('Specific RMSE: %f' % specific_rmse)
        print('Generic RMSE: %f' % generic_rmse)

        return rmse, specific_rmse, generic_rmse

    def evaluate(self):

        if Constants.EVALUATION_METRIC == 'topn_recall':
            return self.evaluate_topn()
        elif Constants.EVALUATION_METRIC == 'rmse':
            return self.evaluate_rmse()
        else:
            raise ValueError('Unrecognized evaluation metric')

    def perform_cross_validation(self, records):

        print(Constants._properties)

        # self.plant_seeds()

        metric_list = []
        specific_metric_list = []
        generic_metric_list = []
        total_cycle_time = 0.0
        num_cycles = Constants.NUM_CYCLES
        num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
        total_iterations = num_cycles * num_folds
        split = 1 - (1/float(num_folds))

        # self.load()

        for i in range(num_cycles):

            print('\n\nCycle: %d/%d' % ((i+1), num_cycles))

            self.records = copy.deepcopy(records)
            if Constants.SHUFFLE_DATA:
                self.shuffle(self.records)

            for j in range(num_folds):

                fold_start = time.time()
                cv_start = float(j) / num_folds
                print('\nFold: %d/%d' % ((j+1), num_folds))

                self.create_tmp_file_names()
                self.train_records, self.test_records = \
                    ETLUtils.split_train_test_copy(
                        self.records, split=split, start=cv_start)
                # subsample_size = int(len(self.train_records)*0.5)
                # self.train_records = self.train_records[:subsample_size]
                self.get_records_to_predict()
                if Constants.USE_CONTEXT:
                    lda_based_context = self.train_topic_model(i, j)
                    self.find_reviews_topics(lda_based_context)
                self.predict()
                metric, specific_metric, generic_metric = self.evaluate()
                metric_list.append(metric)
                specific_metric_list.append(specific_metric)
                generic_metric_list.append(generic_metric)
                print('Accumulated ' + Constants.EVALUATION_METRIC + ': %f' %
                      numpy.mean(metric_list))

                fold_end = time.time()
                fold_time = fold_end - fold_start
                total_cycle_time += fold_time
                self.clear()
                print("Total fold %d time = %f seconds" % ((j+1), fold_time))

        metric_name = Constants.EVALUATION_METRIC
        metric_average = numpy.mean(metric_list)
        metric_stdev = numpy.std(metric_list)
        average_specific_metric = numpy.mean(specific_metric_list)
        average_generic_metric = numpy.mean(generic_metric_list)
        average_cycle_time = total_cycle_time / total_iterations
        print('average %s: %f' % (metric_name, metric_average))
        print(
            'average specific %s: %f' % (metric_name, average_specific_metric))
        print('average generic %s: %f' % (metric_name, average_generic_metric))
        print('standard deviation %s: %f (%f%%)' %
              (metric_name, metric_stdev, (metric_stdev/metric_average*100)))
        print('average cycle time: %f' % average_cycle_time)
        print('End: %s' % time.strftime("%Y/%m/%d-%H:%M:%S"))
        #
        results = copy.deepcopy(Constants._properties)
        results[Constants.EVALUATION_METRIC] = metric_average
        results['specific_' + metric_name] = average_specific_metric
        results['generic_' + metric_name] = average_generic_metric
        results[metric_name + '_stdev'] = metric_stdev
        results['cycle_time'] = average_cycle_time
        results['timestamp'] = time.strftime("%Y/%m/%d-%H:%M:%S")

        write_results_to_csv(results)
        write_results_to_json(results)

        return results

    def run(self):

        self.plant_seeds()
        self.load()

        records = self.original_records

        if Constants.CROSS_VALIDATION_STRATEGY == 'nested_validate':
            num_folds = Constants.CROSS_VALIDATION_NUM_FOLDS
            cycle = Constants.NESTED_CROSS_VALIDATION_CYCLE
            split = 1 - (1/float(num_folds))
            cv_start = float(cycle) / num_folds
            print('cv_start', cv_start)
            records, _ = ETLUtils.split_train_test(
                self.original_records, split, cv_start)
            return self.perform_cross_validation(records)
        elif Constants.CROSS_VALIDATION_STRATEGY == 'nested_test':
            return self.perform_cross_validation(records)
        else:
            raise ValueError('Unknown cross-validation strategy')


# start = time.time()
# my_context_top_n_runner = ContextTopNRunner()
# my_context_top_n_runner.run()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
