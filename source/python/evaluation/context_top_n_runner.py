from subprocess import call
import time
import cPickle as pickle
from etl import ETLUtils
from etl.context_data_converter import ContextDataConverter
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
from topicmodeling.context.reviews_classifier import ReviewsClassifier

__author__ = 'fpena'




my_i = 270
SPLIT_PERCENTAGE = '80'
DATASET = 'hotel'
# my_i = 1000
# SPLIT_PERCENTAGE = '98'
# DATASET = 'restaurant'
DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/'
LIBFM_FOLDER = '/Users/fpena/tmp/libfm-1.42.src/bin/'
GENERATED_FOLDER = DATASET_FOLDER + 'generated_context/'
RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' + DATASET + 's_shuffled_tagged.json'
TRAIN_RECORDS_FILE = RECORDS_FILE + '_train'
TEST_RECORDS_FILE = RECORDS_FILE + '_test'
RECORDS_TO_PREDICT_FILE = GENERATED_FOLDER + 'records_to_predict_' + DATASET + '.json'
REVIEWS_TO_PREDICT_FILE = GENERATED_FOLDER + 'reviews_to_predict_' + DATASET + '.pkl'

CACHE_FOLDER = DATASET_FOLDER + 'cache_context/'
USER_ITEM_MAP_FILE = CACHE_FOLDER + DATASET + '_user_item_map.pkl'
TRAIN_REVIEWS_FILE = CACHE_FOLDER + 'train_reviews_' + DATASET + '.pkl'
TEST_REVIEWS_FILE = CACHE_FOLDER + 'test_reviews_' + DATASET + '.pkl'


def main_split():

    split_command = DATASET_FOLDER + 'split_file.sh'

    command = [
        split_command,
        RECORDS_FILE,
        RECORDS_FILE,
        SPLIT_PERCENTAGE
    ]

    call(command)


def main_context_export():
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    print('num_records', len(records))

    test_records = ETLUtils.load_json_file(TEST_RECORDS_FILE)
    # test_reviews = review_metrics_extractor.build_reviews(test_records)
    # with open(TEST_REVIEWS_FILE, 'wb') as write_file:
    #     pickle.dump(test_reviews, write_file, pickle.HIGHEST_PROTOCOL)
    with open(USER_ITEM_MAP_FILE, 'rb') as read_file:
        user_item_map = pickle.load(read_file)
    with open(TEST_REVIEWS_FILE, 'rb') as read_file:
        test_reviews = pickle.load(read_file)

    top_n_evaluator = TopNEvaluator(records, test_records, DATASET, 10, I, test_reviews)
    top_n_evaluator.initialize(user_item_map)
    top_n_evaluator.export_records_to_predict(
        RECORDS_TO_PREDICT_FILE, REVIEWS_TO_PREDICT_FILE)


def main_lda():

    my_tagged_records_file = DATASET_FOLDER + 'classified_' + DATASET + '_reviews.json'
    my_tagged_reviews_file = DATASET_FOLDER + 'classified_' + DATASET + '_reviews.pkl'

    my_tagged_records = ETLUtils.load_json_file(my_tagged_records_file)
    with open(my_tagged_reviews_file, 'rb') as read_file:
        my_tagged_reviews = pickle.load(read_file)

    my_reviews_classifier = ReviewsClassifier()
    my_reviews_classifier.train(my_tagged_records, my_tagged_reviews)

    print('Trained classifier: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    my_data_preparer = ContextDataConverter(my_reviews_classifier)

    train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
    records_to_predict = ETLUtils.load_json_file(RECORDS_TO_PREDICT_FILE)

    # train_reviews = review_metrics_extractor.build_reviews(train_records)
    # with open(TRAIN_REVIEWS_FILE, 'wb') as write_file:
    #     pickle.dump(train_reviews, write_file, pickle.HIGHEST_PROTOCOL)
    with open(TRAIN_REVIEWS_FILE, 'rb') as read_file:
        train_reviews = pickle.load(read_file)

    # reviews_to_predict = review_metrics_extractor.build_reviews(records_to_predict)
    with open(REVIEWS_TO_PREDICT_FILE, 'rb') as read_file:
        reviews_to_predict = pickle.load(read_file)

    with open(REVIEWS_TO_PREDICT_FILE, 'wb') as write_file:
        pickle.dump(reviews_to_predict, write_file, pickle.HIGHEST_PROTOCOL)

    my_data_preparer.run(
        DATASET, GENERATED_FOLDER, train_records, records_to_predict,
        train_reviews, reviews_to_predict
    )


def main_context_libfm():

    no_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.no_context.libfm'
    no_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_test5.csv.no_context.libfm'
    no_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_no_context.txt'
    no_context_log_file = GENERATED_FOLDER + DATASET + '_no_context.log'
    run_libfm(
        no_context_train_file, no_context_test_file,
        no_context_predictions_file, no_context_log_file)

    context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.context.libfm'
    context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_test5.csv.context.libfm'
    context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_context.txt'
    context_log_file = GENERATED_FOLDER + DATASET + '_context.log'
    run_libfm(
        context_train_file, context_test_file, context_predictions_file,
        context_log_file)

    specific_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.context.libfm'
    specific_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_specific_test5.csv.context.libfm'
    specific_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_specific_context.txt'
    specific_context_log_file = GENERATED_FOLDER + DATASET + '_specific_context.log'
    # run_libfm(
    #     specific_context_train_file, specific_context_test_file,
    #     specific_context_predictions_file, specific_context_log_file)

    specific_no_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.no_context.libfm'
    specific_no_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_specific_test5.csv.no_context.libfm'
    specific_no_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_specific_no_context.txt'
    specific_no_context_log_file = GENERATED_FOLDER + DATASET + '_specific_no_context.log'
    # run_libfm(
    #     specific_no_context_train_file, specific_no_context_test_file,
    #     specific_no_context_predictions_file, specific_no_context_log_file)

    generic_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.context.libfm'
    generic_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_generic_test5.csv.context.libfm'
    generic_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_generic_context.txt'
    generic_context_log_file = GENERATED_FOLDER + DATASET + '_generic_context.log'
    # run_libfm(
    #     generic_context_train_file, generic_context_test_file,
    #     generic_context_predictions_file, generic_context_log_file)

    generic_no_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.no_context.libfm'
    generic_no_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_generic_test5.csv.no_context.libfm'
    generic_no_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_generic_no_context.txt'
    generic_no_context_log_file = GENERATED_FOLDER + DATASET + '_generic_no_context.log'
    # run_libfm(
    #     generic_no_context_train_file, generic_no_context_test_file,
    #     generic_no_context_predictions_file, generic_no_context_log_file)


def run_libfm(train_file, test_file, predictions_file, log_file):

    libfm_command = LIBFM_FOLDER + 'libfm'

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
    ]

    f = open(log_file, "w")
    call(command, stdout=f)


def main_context_evaluate():
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    # print('num_records', len(records))

    test_file = RECORDS_FILE + '_test'
    test_records = ETLUtils.load_json_file(test_file)

    top_n_evaluator = TopNEvaluator(records, test_records, DATASET, 10, I)
    top_n_evaluator.calculate_important_items()
    # top_n_evaluator.initialize()

    # records_to_predict_file = DATASET_FOLDER + 'generated/records_to_predict_' + DATASET + '.json'
    top_n_evaluator.load_records_to_predict(RECORDS_TO_PREDICT_FILE)

    predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_no_context.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    print('No context recall: %f' % top_n_evaluator.recall)

    predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_context.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    print('Context recall: %f' % top_n_evaluator.recall)
    context_recall = top_n_evaluator.recall

    # predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_specific_no_context.txt'
    # predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    # top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    # print('Specific no context recall: %f' % top_n_evaluator.recall)

    # predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_specific_context.txt'
    # predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    # top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    # print('Specific context recall: %f' % top_n_evaluator.recall)

    # predictions_file = GENERATED_FOLDER +  'predictions_' + DATASET + '_generic_no_context.txt'
    # predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    # top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    # print('Generic no context recall: %f' % top_n_evaluator.recall)

    # predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_generic_context.txt'
    # predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    # top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    # print('Generic context recall: %f' % top_n_evaluator.recall)

    predictions_file = DATASET_FOLDER + '/generated_plain/predictions_' + DATASET + '.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    print('Plain recall: %f' % top_n_evaluator.recall)

    return context_recall


def super_main_lda():

    total_recall = 0.0
    num_iterations = 10

    for i in range(num_iterations):
        print('\nCycle: %d' % i)

        print('main split')
        main_split()
        print('main export')
        main_context_export()
        print('main converter')
        main_lda()
        print('main libfm')
        main_context_libfm()
        print('main evaluate')
        total_recall += main_context_evaluate()

    average_recall = total_recall / num_iterations
    print('average_recall', average_recall)


def experiment():
    records = ETLUtils.load_json_file(RECORDS_FILE)
    # print('num_records', len(records))

    test_file = RECORDS_FILE + '_test'
    test_records = ETLUtils.load_json_file(test_file)

    top_n_evaluator = TopNEvaluator(records, test_records, DATASET, 10, my_i)
    top_n_evaluator.calculate_important_items()

    # records_to_predict_file = DATASET_FOLDER + 'generated_plain/' + 'records_to_predict_' + DATASET + '.json'
    records_to_predict_file = DATASET_FOLDER + 'generated_context/' + 'records_to_predict_' + DATASET + '.json'
    top_n_evaluator.load_records_to_predict(records_to_predict_file)

    predictions_file = DATASET_FOLDER + 'generated_plain/' + 'predictions_' + DATASET + '.txt'
    # predictions_file = DATASET_FOLDER + 'generated_context/' + 'predictions_' + DATASET + '_no_context.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    print('No context recall: %f' % top_n_evaluator.recall)



start = time.time()
# main()
# main_split()
# main_context_export()
# main_lda()
# main_context_libfm()
# main_context_evaluate()
super_main_lda()
# experiment()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
