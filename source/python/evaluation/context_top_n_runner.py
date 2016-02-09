from subprocess import call
import time
import cPickle as pickle
from etl import ETLUtils
from etl import libfm_converter
from evaluation import rmse_calculator
from evaluation.top_n_evaluator import TopNEvaluator
from topicmodeling.context.lda_based_context import LdaBasedContext
from topicmodeling.context.reviews_classifier import ReviewsClassifier
from tripadvisor.fourcity import extractor
from utils import constants

__author__ = 'fpena'


my_i = 270
SPLIT_PERCENTAGE = '80'
DATASET = 'hotel'
# my_i = 1000
# SPLIT_PERCENTAGE = '98'
# DATASET = 'restaurant'
REVIEW_TYPE = ''
# REVIEW_TYPE = 'specific'
# REVIEW_TYPE = 'generic'

# Folders
DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/stuff2/'
LIBFM_FOLDER = '/Users/fpena/tmp/libfm-master/bin/'
GENERATED_FOLDER = DATASET_FOLDER + 'generated_context/'

# Main Files
CACHE_FOLDER = DATASET_FOLDER + 'cache_context/'
RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
               DATASET + 's_shuffled_tagged.json'
# TRAIN_RECORDS_FILE = RECORDS_FILE + '_train'
TRAIN_RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
               DATASET + 's_shuffled_tagged.json_train'
TEST_RECORDS_FILE = RECORDS_FILE + '_test'
TAGGED_RECORDS_FILE = DATASET_FOLDER + 'classified_' + DATASET + '_reviews.json'
TAGGED_REVIEWS_FILE = DATASET_FOLDER + 'classified_' + DATASET + '_reviews.pkl'

# Generated files
RECORDS_TO_PREDICT_FILE = GENERATED_FOLDER +\
                          'records_to_predict_' + DATASET + '.json'

# Cache files
USER_ITEM_MAP_FILE = CACHE_FOLDER + DATASET + '_' + 'user_item_map.pkl'
REVIEWS_FILE = DATASET_FOLDER + 'reviews_' + DATASET + '_shuffled.pkl'
TRAIN_REVIEWS_FILE = CACHE_FOLDER +\
                     'train_reviews_' + DATASET + '.pkl'
# TRAIN_REVIEWS_FILE = CACHE_FOLDER + REVIEW_TYPE +\
#                      'train_reviews_' + DATASET + '.pkl'
TEST_REVIEWS_FILE = CACHE_FOLDER + 'test_reviews_' + DATASET + '.pkl'
TOPIC_MODEL_FILE = CACHE_FOLDER + 'topic_model_' + DATASET + '.pkl'


def split_binary_reviews():

    train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
    test_records = ETLUtils.load_json_file(TEST_RECORDS_FILE)
    records = ETLUtils.load_json_file(RECORDS_FILE)
    num_train_records = len(train_records)

    with open(REVIEWS_FILE, 'rb') as read_file:
        reviews = pickle.load(read_file)

    train_reviews = []
    for record, review in zip(records, reviews)[:num_train_records]:
        review.user_id = record[constants.USER_ID_FIELD]
        review.item_id = record[constants.ITEM_ID_FIELD]
        review.rating = record[constants.RATING_FIELD]
        train_reviews.append(review)

    test_reviews = []
    for record, review in zip(records, reviews)[num_train_records:]:
        review.user_id = record[constants.USER_ID_FIELD]
        review.item_id = record[constants.ITEM_ID_FIELD]
        review.rating = record[constants.RATING_FIELD]
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

    print(len(train_records), len(train_reviews))
    print(len(test_records), len(test_reviews))

    for record, review in zip(train_records, train_reviews):
        if record[constants.TEXT_FIELD] != review.text:
            print('Something went wrong...')

    for record, review in zip(test_records, test_reviews):
        if record[constants.TEXT_FIELD] != review.text:
            print('Something went wrong...')


def build_headers(context_rich_topics):
    headers = [
        constants.RATING_FIELD,
        constants.USER_ID_FIELD,
        constants.ITEM_ID_FIELD
    ]
    for topic in context_rich_topics:
        topic_id = 'topic' + str(topic[0])
        headers.append(topic_id)
    return headers


def create_user_item_map(records):
    user_ids = extractor.get_groupby_list(records, constants.USER_ID_FIELD)
    user_item_map = {}
    user_count = 0

    for user_id in user_ids:
        user_records =\
            ETLUtils.filter_records(records, constants.USER_ID_FIELD, [user_id])
        user_items =\
            extractor.get_groupby_list(user_records, constants.ITEM_ID_FIELD)
        user_item_map[user_id] = user_items
        user_count += 1

        # print("user count %d" % user_count),
        print 'user count: {0}\r'.format(user_count),

    print

    return user_item_map


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
        predictions_file,
        '-seed',
        '0'
    ]

    f = open(log_file, "w")
    call(command, stdout=f)


def split():

    print('main split: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    split_command = DATASET_FOLDER + 'split_file.sh'

    command = [
        split_command,
        RECORDS_FILE,
        RECORDS_FILE,
        SPLIT_PERCENTAGE
    ]

    call(command)

    # split_binary_reviews()


def export():
    print('export: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    print('num_records', len(records))

    test_records = ETLUtils.load_json_file(TEST_RECORDS_FILE)

    if REVIEW_TYPE:
        # records = ETLUtils.filter_records(
        #     records, constants.PREDICTED_CLASS_FIELD, [REVIEW_TYPE])
        test_records = ETLUtils.filter_records(
            test_records, constants.PREDICTED_CLASS_FIELD, [REVIEW_TYPE]
        )

    user_item_map = create_user_item_map(records)

    with open(USER_ITEM_MAP_FILE, 'wb') as write_file:
        pickle.dump(user_item_map, write_file, pickle.HIGHEST_PROTOCOL)

    with open(USER_ITEM_MAP_FILE, 'rb') as read_file:
        user_item_map = pickle.load(read_file)

    top_n_evaluator =\
        TopNEvaluator(records, test_records, DATASET, 10, I)
    top_n_evaluator.initialize(user_item_map)
    top_n_evaluator.export_records_to_predict(RECORDS_TO_PREDICT_FILE)


def train_topic_model():
    print('train topic model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    # train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
    #
    # lda_based_context = LdaBasedContext(train_records)
    # lda_based_context.get_context_rich_topics()
    #
    # print('Trained LDA Model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    #
    # with open(TOPIC_MODEL_FILE, 'wb') as write_file:
    #     pickle.dump(lda_based_context, write_file, pickle.HIGHEST_PROTOCOL)

    with open(TOPIC_MODEL_FILE, 'rb') as read_file:
        lda_based_context = pickle.load(read_file)

    return lda_based_context


def find_reviews_topics(lda_based_context):
    print('find topics: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    train_records = ETLUtils.load_json_file(TRAIN_RECORDS_FILE)
    records_to_predict = ETLUtils.load_json_file(RECORDS_TO_PREDICT_FILE)

    contextual_train_set =\
        lda_based_context.find_contextual_topics(train_records)
    contextual_test_set =\
        lda_based_context.find_contextual_topics(records_to_predict)

    print('contextual test set size: %d' % len(contextual_test_set))

    headers = build_headers(lda_based_context.context_rich_topics)

    print('Exported contextual topics: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    return contextual_train_set, contextual_test_set, headers


def filter_reviews(records, reviews, review_type):
    print('filter: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    if not review_type:
        return records, reviews

    filtered_records = []
    filtered_reviews = []

    for record, review in zip(records, reviews):
        if record[constants.PREDICTED_CLASS_FIELD] == review_type:
            filtered_records.append(record)
            filtered_reviews.append(review)

    return filtered_records, filtered_reviews


def prepare(contextual_train_set, contextual_test_set, headers):
    print('prepare: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    contextual_train_set =\
        ETLUtils.select_fields(headers, contextual_train_set)
    contextual_test_set =\
        ETLUtils.select_fields(headers, contextual_test_set)

    csv_train_file =\
        GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_train5.csv'
    csv_test_file =\
        GENERATED_FOLDER + 'yelp_' + DATASET + '_context_shuffled_test5.csv'

    ETLUtils.save_csv_file(csv_train_file, contextual_train_set, headers)
    ETLUtils.save_csv_file(csv_test_file, contextual_test_set, headers)

    print('Exported CSV and JSON files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    csv_files = [
        csv_train_file,
        csv_test_file
    ]

    num_cols = len(headers)
    context_cols = num_cols
    print('num_cols', num_cols)
    # print('context_cols', context_cols)

    libfm_converter.csv_to_libfm(
        csv_files, 0, [1, 2], range(3, context_cols), ',', has_header=True,
        suffix='.no_context.libfm')
    libfm_converter.csv_to_libfm(
        csv_files, 0, [1, 2], [], ',', has_header=True,
        suffix='.context.libfm')

    print('Exported LibFM files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))


def predict():
    print('predict: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

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


def evaluate():
    print('evaluate: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    I = my_i

    records = ETLUtils.load_json_file(RECORDS_FILE)
    # print('num_records', len(records))

    test_file = RECORDS_FILE + '_test'
    test_records = ETLUtils.load_json_file(test_file)

    top_n_evaluator = TopNEvaluator(records, test_records, DATASET, 10, I)
    top_n_evaluator.find_important_records()
    # top_n_evaluator.initialize()

    top_n_evaluator.load_records_to_predict(RECORDS_TO_PREDICT_FILE)

    predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_no_context.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    top_n_evaluator.evaluate(predictions)
    no_context_recall = top_n_evaluator.recall

    predictions_file = GENERATED_FOLDER + 'predictions_' + DATASET + '_context.txt'
    predictions = rmse_calculator.read_targets_from_txt(predictions_file)
    top_n_evaluator.evaluate(predictions)
    context_recall = top_n_evaluator.recall

    print('No context recall: %f' % no_context_recall)
    print('Context recall: %f' % context_recall)

    return context_recall, no_context_recall


def super_main_lda():

    total_context_recall = 0.0
    total_no_context_recall = 0.0
    total_cycle_time = 0.0
    num_iterations = 1

    # split()
    # export()

    for i in range(num_iterations):
        cycle_start = time.time()
        print('\nCycle: %d' % i)

        lda_based_context = train_topic_model()
        contextual_train_set, contextual_test_set, headers =\
            find_reviews_topics(lda_based_context)
        prepare(contextual_train_set, contextual_test_set, headers)
        predict()
        context_recall, no_context_recall = evaluate()
        total_context_recall += context_recall
        total_no_context_recall += no_context_recall

        cycle_end = time.time()
        cycle_time = cycle_end - cycle_start
        total_cycle_time += cycle_time
        print("Total cycle %d time = %f seconds" % (i, cycle_time))

    average_context_recall = total_context_recall / num_iterations
    average_no_context_recall = total_no_context_recall / num_iterations
    average_cycle_time = total_cycle_time / num_iterations
    improvement = (average_context_recall / average_no_context_recall - 1) * 100
    print('average no context recall', average_no_context_recall)
    print('average context recall', average_context_recall)
    print('average improvement: %f2.3%%' % improvement)
    print('average cycle time', average_cycle_time)
    print('End: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))


def experiment():

    reviews_file = DATASET_FOLDER + 'reviews_' + DATASET + '_shuffled.pkl'
    records_file = DATASET_FOLDER + 'yelp_training_set_review_' + DATASET + 's_shuffled_tagged.json'
    with open(reviews_file, 'rb') as read_file:
        reviews = pickle.load(read_file)

    records = ETLUtils.load_json_file(records_file)

    specific_records = []
    generic_records = []
    specific_reviews = []
    generic_reviews = []

    index = 0

    for record in records:
        if record[constants.PREDICTED_CLASS_FIELD] == 'specific':
            specific_records.append(record)
            specific_reviews.append(reviews[index])
        if record[constants.PREDICTED_CLASS_FIELD] == 'generic':
            generic_records.append(record)
            generic_reviews.append(reviews[index])
        index += 1

    print('reviews length', len(reviews))
    print('records length', len(records))
    print('specific reviews length', len(specific_reviews))
    print('specific records length', len(specific_records))
    print('generic reviews length', len(generic_reviews))
    print('generic records length', len(generic_reviews))

    specific_records_file = DATASET_FOLDER + 'specific_yelp_training_set_review_' + DATASET + 's_shuffled_tagged.json'
    generic_records_file = DATASET_FOLDER + 'generic_yelp_training_set_review_' + DATASET + 's_shuffled_tagged.json'
    specific_reviews_file = DATASET_FOLDER + 'specific_reviews_' + DATASET + '_shuffled.pkl'
    generic_reviews_file = DATASET_FOLDER + 'generic_reviews_' + DATASET + '_shuffled.pkl'
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



start = time.time()
# main()
# split()
# export()
# main_lda()
# predict()
# evaluate()
super_main_lda()
# experiment()
# split_binary_reviews()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
