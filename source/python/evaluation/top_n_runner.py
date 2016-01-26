from subprocess import call
import time
from etl import ETLUtils
from etl.libfm_converter import csv_to_libfm
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
from topicmodeling.context.lda_based_context import LdaBasedContext
import cPickle as pickle

__author__ = 'fpena'


my_i = 270
SPLIT_PERCENTAGE = '80'
DATASET = 'hotel'
# DATASET = 'restaurant'
DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/'
RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' + DATASET + 's_shuffled_tagged.json'


def main_split():

    split_command = DATASET_FOLDER + 'split_file.sh'

    command = [
        split_command,
        RECORDS_FILE,
        RECORDS_FILE,
        SPLIT_PERCENTAGE
    ]

    call(command)


def main_export():
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    print('num_records', len(records))

    test_file = RECORDS_FILE + '_test'
    test_records = ETLUtils.load_json_file(test_file)

    top_n_evaluator = TopNEvaluator(records, test_records, 10, I)
    top_n_evaluator.initialize()
    # top_n_evaluator.get_records_to_predict()

    records_to_predict_file = DATASET_FOLDER + 'generated/records_to_predict_' + DATASET + '.json'
    top_n_evaluator.export_records_to_predict(records_to_predict_file)


def main_load():
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    # print('num_records', len(records))

    test_file = RECORDS_FILE + '_test'
    test_records = ETLUtils.load_json_file(test_file)

    top_n_evaluator = TopNEvaluator(records, test_records, 10, I)
    top_n_evaluator.important_items =\
        TopNEvaluator.calculate_important_items(test_records)
    # top_n_evaluator.initialize()

    records_to_predict_file = DATASET_FOLDER + 'generated/records_to_predict_' + DATASET + '.json'
    top_n_evaluator.load_records_to_predict(records_to_predict_file)

    predictions_file = '/Users/fpena/tmp/libfm-1.42.src/bin/predictions_' + DATASET + '_no_context.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)

    # print('total predictions', len(predictions))
    top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    print('recall', top_n_evaluator.recall)

    return top_n_evaluator.recall


def main_converter():

    json_file1 = RECORDS_FILE + '_train'
    json_file2 = DATASET_FOLDER + 'generated/' + 'records_to_predict_' + DATASET + '.json'

    export_folder = '/Users/fpena/tmp/libfm-1.42.src/bin/'
    export_file = export_folder + 'yelp_training_set_review_' + DATASET + 's_shuffled_train.csv'
    csv_file = export_folder + 'records_to_predict_' + DATASET + '.csv'

    ETLUtils.json_to_csv(json_file1, export_file, 'user_id', 'business_id', 'stars', False, True)
    ETLUtils.json_to_csv(json_file2, csv_file, 'user_id', 'business_id', 'stars', False, True)

    csv_files = [
        export_file,
        csv_file
    ]

    csv_to_libfm(csv_files, 2, [0, 1], [], ',', has_header=True)


def main_libfm():

    folder = '/Users/fpena/tmp/libfm-1.42.src/bin/'
    libfm_command = folder + 'libfm'
    train_file = folder + 'yelp_training_set_review_' + DATASET + 's_shuffled_train.csv.libfm'
    test_file = folder + 'records_to_predict_' + DATASET + '.csv.libfm'
    predictions_file = folder + 'predictions_' + DATASET + '_no_context.txt'
    log_file = folder + DATASET + '_no_context.log'

    command = [
        libfm_command,
        '-task',
        'r',
        '-train',
        train_file,
        '-test',
        test_file,
        '-dim',
        '1,1,8',
        '-out',
        predictions_file
        # '>',
        # log_file
    ]

    f = open(log_file, "w")
    call(command, stdout=f)


def main_lda():

    training_records_file = DATASET_FOLDER + 'yelp_training_set_review_' + DATASET + 's_shuffled_tagged.json'
    training_records = ETLUtils.load_json_file(training_records_file)
    training_reviews_file = DATASET_FOLDER + 'reviews_' + DATASET + '_shuffled.pkl'
    # reviews = context_utils.load_reviews(reviews_file)
    # print("reviews:", len(reviews))
    #
    # reviews = None
    # my_file = '/Users/fpena/tmp/reviews_restaurant_shuffled.pkl'
    # my_file = '/Users/fpena/tmp/sentences_hotel.pkl'
    # my_file = '/Users/fpena/tmp/reviews_hotel.pkl'
    # my_file = '/Users/fpena/tmp/reviews_spa.pkl'

    # with open(my_file, 'wb') as write_file:
    #     pickle.dump(self.reviews, write_file, pickle.HIGHEST_PROTOCOL)
    # training_records_file = DATASET_FOLDER + 'classified_' + dataset + '_reviews.json'
    # training_reviews_file = DATASET_FOLDER + 'classified_' + dataset + '_reviews.pkl'

    with open(training_reviews_file, 'rb') as read_file:
        training_reviews = pickle.load(read_file)

    print('lda num_reviews', len(training_reviews))
    # lda_context_utils.discover_topics(reviews, 150)
    lda_based_context = LdaBasedContext(training_records, training_reviews)
    # lda_based_context.training_set_file = training_records_file
    # lda_based_context.training_reviews_file = training_reviews_file
    lda_based_context.init_reviews()

    topics = lda_based_context.get_context_rich_topics()
    print(topics)
    print('total_topics', len(topics))

    records_file = DATASET_FOLDER + 'classified_' + DATASET + '_reviews.json'
    reviews_file = DATASET_FOLDER + 'classified_' + DATASET + '_reviews.pkl'
    json_file = DATASET_FOLDER + 'yelp_' + DATASET + '_context_shuffled4.json'
    csv_file = DATASET_FOLDER + 'yelp_' + DATASET + '_context_shuffled4.csv'
    lda_based_context.export_contextual_records(records_file, reviews_file, json_file, csv_file)


def super_main():

    total_recall = 0.0
    num_iterations = 10

    for _ in range(num_iterations):
        print('main split')
        main_split()
        print('main export')
        main_export()
        print('main converter')
        main_converter()
        print('main libfm')
        main_libfm()
        print('main load')
        total_recall += main_load()

    average_recall = total_recall / num_iterations
    print('average_recall', average_recall)


start = time.time()
# main()
# main_split()
# main_export()
# main_converter()
# main_libfm()
# main_load()
super_main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)

