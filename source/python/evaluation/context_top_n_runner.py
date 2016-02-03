from subprocess import call
import time
import cPickle as pickle
from etl import ETLUtils
from etl.context_data_converter import ContextDataConverter
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
from topicmodeling.context import review_metrics_extractor
from topicmodeling.context.reviews_classifier import ReviewsClassifier
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


my_i = 270
SPLIT_PERCENTAGE = '80'
ITEM_TYPE = 'hotel'
# my_i = 1000
# SPLIT_PERCENTAGE = '98'
# ITEM_TYPE = 'restaurant'
REVIEW_TYPE = ''
# REVIEW_TYPE = 'specific_'
# REVIEW_TYPE = 'generic_'

# Folders
DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/'
# DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/2nd_generation/'

LIBFM_FOLDER = '/Users/fpena/tmp/libfm-1.42.src/bin/'
GENERATED_FOLDER = DATASET_FOLDER + 'generated_context/'

# Main Files
CACHE_FOLDER = DATASET_FOLDER + 'cache_context/'
RECORDS_FILE = DATASET_FOLDER + REVIEW_TYPE + 'yelp_training_set_review_' +\
               ITEM_TYPE + 's_shuffled_tagged.json'
# RECORDS_FILE = DATASET_FOLDER + 'reviews_' + ITEM_TYPE + '_shuffled.json'
# TRAIN_RECORDS_FILE = RECORDS_FILE + '_train'
TRAIN_RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
               ITEM_TYPE + 's_shuffled_tagged.json_train'
TEST_RECORDS_FILE = RECORDS_FILE + '_test'
TAGGED_RECORDS_FILE = DATASET_FOLDER + 'classified_' + ITEM_TYPE + '_reviews.json'
TAGGED_REVIEWS_FILE = DATASET_FOLDER + 'classified_' + ITEM_TYPE + '_reviews.pkl'

# Generated files
RECORDS_TO_PREDICT_FILE = GENERATED_FOLDER +\
                          'records_to_predict_' + ITEM_TYPE + '.json'
REVIEWS_TO_PREDICT_FILE = GENERATED_FOLDER +\
                          'reviews_to_predict_' + ITEM_TYPE + '.pkl'

# Cache files
USER_ITEM_MAP_FILE = CACHE_FOLDER + ITEM_TYPE + '_' +\
                     REVIEW_TYPE + 'user_item_map.pkl'
TRAIN_REVIEWS_FILE = CACHE_FOLDER +\
                     'train_reviews_' + ITEM_TYPE + '.pkl'
# TRAIN_REVIEWS_FILE = CACHE_FOLDER + REVIEW_TYPE +\
#                      'train_reviews_' + DATASET + '.pkl'
TEST_REVIEWS_FILE = CACHE_FOLDER + REVIEW_TYPE +\
                    'test_reviews_' + ITEM_TYPE + '.pkl'


def main_split():

    split_command = DATASET_FOLDER + 'split_file.sh'

    command = [
        split_command,
        RECORDS_FILE,
        RECORDS_FILE,
        SPLIT_PERCENTAGE
    ]

    call(command)


def create_user_item_map(records):
    user_ids = extractor.get_groupby_list(records, 'user_id')
    user_item_map = {}
    user_count = 0

    for user_id in user_ids:
        user_records =\
            ETLUtils.filter_records(records, 'user_id', [user_id])
        user_items = extractor.get_groupby_list(user_records, 'business_id')
        user_item_map[user_id] = user_items
        user_count += 1

        # print("user count %d" % user_count),
        print 'user count: {0}\r'.format(user_count),

    print

    return user_item_map


def main_context_export():
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    print('num_records', len(records))

    test_records = ETLUtils.load_json_file(TEST_RECORDS_FILE)
    # test_reviews = review_metrics_extractor.build_reviews(test_records)
    # with open(TEST_REVIEWS_FILE, 'wb') as write_file:
    #     pickle.dump(test_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    # user_item_map = create_user_item_map(records)
    #
    # with open(USER_ITEM_MAP_FILE, 'wb') as write_file:
    #     pickle.dump(user_item_map, write_file, pickle.HIGHEST_PROTOCOL)


    with open(USER_ITEM_MAP_FILE, 'rb') as read_file:
        user_item_map = pickle.load(read_file)
    with open(TEST_REVIEWS_FILE, 'rb') as read_file:
        test_reviews = pickle.load(read_file)

    top_n_evaluator =\
        TopNEvaluator(records, test_records, ITEM_TYPE, 10, I, test_reviews)
    top_n_evaluator.initialize(user_item_map)
    top_n_evaluator.export_records_to_predict(
        RECORDS_TO_PREDICT_FILE, REVIEWS_TO_PREDICT_FILE)


def main_lda():

    my_tagged_records = ETLUtils.load_json_file(TAGGED_RECORDS_FILE)
    with open(TAGGED_REVIEWS_FILE, 'rb') as read_file:
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

    # reviews_to_predict =\
    #     review_metrics_extractor.build_reviews(records_to_predict)
    with open(REVIEWS_TO_PREDICT_FILE, 'rb') as read_file:
        reviews_to_predict = pickle.load(read_file)

    # with open(REVIEWS_TO_PREDICT_FILE, 'wb') as write_file:
    #     pickle.dump(reviews_to_predict, write_file, pickle.HIGHEST_PROTOCOL)

    my_data_preparer.run(
        ITEM_TYPE, GENERATED_FOLDER, train_records, records_to_predict,
        train_reviews, reviews_to_predict
    )


def main_context_libfm():

    no_context_train_file = GENERATED_FOLDER + 'yelp_' + ITEM_TYPE + '_context_shuffled_train5.csv.no_context.libfm'
    no_context_test_file = GENERATED_FOLDER + 'yelp_' + ITEM_TYPE + '_context_shuffled_test5.csv.no_context.libfm'
    no_context_predictions_file = GENERATED_FOLDER + 'predictions_' + ITEM_TYPE + '_no_context.txt'
    no_context_log_file = GENERATED_FOLDER + ITEM_TYPE + '_no_context.log'
    run_libfm(
        no_context_train_file, no_context_test_file,
        no_context_predictions_file, no_context_log_file)

    context_train_file = GENERATED_FOLDER + 'yelp_' + ITEM_TYPE + '_context_shuffled_train5.csv.context.libfm'
    context_test_file = GENERATED_FOLDER + 'yelp_' + ITEM_TYPE + '_context_shuffled_test5.csv.context.libfm'
    context_predictions_file = GENERATED_FOLDER + 'predictions_' + ITEM_TYPE + '_context.txt'
    context_log_file = GENERATED_FOLDER + ITEM_TYPE + '_context.log'
    run_libfm(
        context_train_file, context_test_file, context_predictions_file,
        context_log_file)

    # specific_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.context.libfm'
    # specific_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_specific_test5.csv.context.libfm'
    # specific_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_specific_context.txt'
    # specific_context_log_file = GENERATED_FOLDER + DATASET + '_specific_context.log'
    # run_libfm(
    #     specific_context_train_file, specific_context_test_file,
    #     specific_context_predictions_file, specific_context_log_file)
    #
    # specific_no_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.no_context.libfm'
    # specific_no_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_specific_test5.csv.no_context.libfm'
    # specific_no_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_specific_no_context.txt'
    # specific_no_context_log_file = GENERATED_FOLDER + DATASET + '_specific_no_context.log'
    # run_libfm(
    #     specific_no_context_train_file, specific_no_context_test_file,
    #     specific_no_context_predictions_file, specific_no_context_log_file)
    #
    # generic_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.context.libfm'
    # generic_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_generic_test5.csv.context.libfm'
    # generic_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_generic_context.txt'
    # generic_context_log_file = GENERATED_FOLDER + DATASET + '_generic_context.log'
    # run_libfm(
    #     generic_context_train_file, generic_context_test_file,
    #     generic_context_predictions_file, generic_context_log_file)
    #
    # generic_no_context_train_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv.no_context.libfm'
    # generic_no_context_test_file = GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_generic_test5.csv.no_context.libfm'
    # generic_no_context_predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_generic_no_context.txt'
    # generic_no_context_log_file = GENERATED_FOLDER + DATASET + '_generic_no_context.log'
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

    top_n_evaluator = TopNEvaluator(records, test_records, ITEM_TYPE, 10, I)
    top_n_evaluator.calculate_important_items()
    # top_n_evaluator.initialize()

    # records_to_predict_file = DATASET_FOLDER + 'generated/records_to_predict_' + DATASET + '.json'
    top_n_evaluator.load_records_to_predict(RECORDS_TO_PREDICT_FILE)

    predictions_file = GENERATED_FOLDER + 'predictions_' + ITEM_TYPE + '_no_context.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # print('total predictions', len(predictions))
    top_n_evaluator.evaluate(predictions)
    # print('precision', top_n_evaluator.precision)
    print('No context recall: %f' % top_n_evaluator.recall)

    predictions_file = GENERATED_FOLDER + 'predictions_' + ITEM_TYPE + '_context.txt'
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

    # predictions_file = DATASET_FOLDER + 'generated_plain/predictions_' + DATASET + '.txt'
    # predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    # # print('total predictions', len(predictions))
    # top_n_evaluator.evaluate(predictions)
    # # print('precision', top_n_evaluator.precision)
    # print('Plain recall: %f' % top_n_evaluator.recall)

    return context_recall


def super_main_lda():

    total_recall = 0.0
    num_iterations = 10

    for i in range(num_iterations):
        print('\nCycle: %d' % i)

        print('main split: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        # main_split()
        print('main export: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        main_context_export()
        print('main converter: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        main_lda()
        print('main libfm: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        main_context_libfm()
        print('main evaluate: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
        total_recall += main_context_evaluate()

    average_recall = total_recall / num_iterations
    print('average_recall', average_recall)
    print('End: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))


def experiment():

    reviews_file = DATASET_FOLDER + 'reviews_' + ITEM_TYPE + '_shuffled.pkl'
    records_file = DATASET_FOLDER + 'yelp_training_set_review_' + ITEM_TYPE + 's_shuffled_tagged.json'
    with open(reviews_file, 'rb') as read_file:
        reviews = pickle.load(read_file)

    records = ETLUtils.load_json_file(records_file)

    specific_records = []
    generic_records = []
    specific_reviews = []
    generic_reviews = []

    index = 0

    for record in records:
        if record['predicted_class'] == 'specific':
            specific_records.append(record)
            specific_reviews.append(reviews[index])
        if record['predicted_class'] == 'generic':
            generic_records.append(record)
            generic_reviews.append(reviews[index])
        index += 1

    print('reviews length', len(reviews))
    print('records length', len(records))
    print('specific reviews length', len(specific_reviews))
    print('specific records length', len(specific_records))
    print('generic reviews length', len(generic_reviews))
    print('generic records length', len(generic_reviews))

    specific_records_file = DATASET_FOLDER + 'specific_yelp_training_set_review_' + ITEM_TYPE + 's_shuffled_tagged.json'
    generic_records_file = DATASET_FOLDER + 'generic_yelp_training_set_review_' + ITEM_TYPE + 's_shuffled_tagged.json'
    specific_reviews_file = DATASET_FOLDER + 'specific_reviews_' + ITEM_TYPE + '_shuffled.pkl'
    generic_reviews_file = DATASET_FOLDER + 'generic_reviews_' + ITEM_TYPE + '_shuffled.pkl'
    with open(specific_reviews_file, 'wb') as write_file:
        pickle.dump(specific_reviews, write_file, pickle.HIGHEST_PROTOCOL)
    with open(generic_reviews_file, 'wb') as write_file:
        pickle.dump(generic_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    # print(generic_records[10])
    # print(generic_records[100])

    # print('dumped files')

    # ETLUtils.save_json_file(generic_records_file, generic_records)
    # print('saved generic JSON')
    # ETLUtils.save_json_file(specific_records_file, specific_records)
    # print('saved specific JSON')

    print('saved JSON files')


def tmp_function():

    # training_records_file = DATASET_FOLDER + REVIEW_TYPE +\
    #                         'yelp_training_set_review_' + DATASET + 's_shuffled_tagged.json_train'
    # train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
    # test_records = ETLUtils.load_json_file(TEST_RECORDS_FILE)
    # reviews_file = DATASET_FOLDER + 'reviews_' + DATASET + '_shuffled.pkl'
    # with open(reviews_file, 'rb') as read_file:
    #     reviews = pickle.load(read_file)
    #
    # train_reviews = []
    # test_reviews = []
    #
    # num_train_records = len(train_records)
    #
    # print('training records length', num_train_records)
    #
    # for record, review in zip(train_records, reviews):
    #     review.user_id = record['user_id']
    #     review.item_id = record['business_id']
    #     review.rating = record['stars']
    #     train_reviews.append(review)
    #
    # print('training reviews length', len(train_reviews))
    #
    # for record, review in zip(test_records, reviews[num_train_records:]):
    #     review.user_id = record['user_id']
    #     review.item_id = record['business_id']
    #     review.rating = record['stars']
    #     test_reviews.append(review)
    #
    # with open(TEST_REVIEWS_FILE, 'wb') as write_file:
    #     pickle.dump(train_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
    test_records = ETLUtils.load_json_file(TEST_RECORDS_FILE)
    records = ETLUtils.load_json_file(RECORDS_FILE)
    num_train_records = len(train_records)

    print(TRAIN_REVIEWS_FILE)

    reviews_file = DATASET_FOLDER + REVIEW_TYPE + 'reviews_' + ITEM_TYPE + '_shuffled.pkl'
    print(reviews_file)

    with open(reviews_file, 'rb') as read_file:
        reviews = pickle.load(read_file)

    train_reviews = []
    for record, review in zip(records, reviews)[:num_train_records]:
        review.user_id = record['user_id']
        review.item_id = record['business_id']
        review.rating = record['stars']
        train_reviews.append(review)

    test_reviews = []
    for record, review in zip(records, reviews)[num_train_records:]:
        review.user_id = record['user_id']
        review.item_id = record['business_id']
        review.rating = record['stars']
        test_reviews.append(review)
    #
    print('TRAIN_REVIEWS_FILE', TRAIN_REVIEWS_FILE)
    print('TEST_REVIEWS_FILE', TEST_REVIEWS_FILE)

    with open(TRAIN_REVIEWS_FILE, 'wb') as write_file:
        pickle.dump(train_reviews, write_file, pickle.HIGHEST_PROTOCOL)
    with open(TEST_REVIEWS_FILE, 'wb') as write_file:
        pickle.dump(test_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    with open(TRAIN_REVIEWS_FILE, 'rb') as read_file:
        train_reviews = pickle.load(read_file)
    with open(TEST_REVIEWS_FILE, 'rb') as read_file:
        test_reviews = pickle.load(read_file)

    # print(train_records[0]['text'])
    # print('\n--------------------------------\n')
    # print(train_reviews[0].text)
    # print('\n\n********************************\n\n')
    # print(train_records[100]['text'])
    # print('\n--------------------------------\n')
    # print(train_reviews[100].text)
    # print('\n\n********************************\n\n')
    # print(train_records[101]['text'])
    # print('\n--------------------------------\n')
    # print(train_reviews[101].text)
    # print('\n\n********************************\n\n')

    # print(train_reviews[0].text)
    # print('\n\n********************************\n\n')
    # print(train_reviews[1].text)
    # print('\n\n********************************\n\n')
    # print(train_reviews[2].text)
    # print('\n\n********************************\n\n')

    print(len(train_records), len(train_reviews))
    print(len(test_records), len(test_reviews))

    for record, review in zip(train_records, train_reviews):
        if record['text'] != review.text:
            print('Something went wrong...')

    for record, review in zip(test_records, test_reviews):
        if record['text'] != review.text:
            print('Something went wrong...')






start = time.time()
# main()
# main_split()
# main_context_export()
# main_lda()
# main_context_libfm()
# main_context_evaluate()
super_main_lda()
# experiment()
# tmp_function()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
