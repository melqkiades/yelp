# from topicmodeling.hiddenfactortopics.topic_corpus import TopicCorpus
import codecs
import csv
import itertools

from scipy.stats import wilcoxon, ttest_ind
from sklearn import decomposition

import langdetect
import pandas
from scipy.io import mmread
from scipy.sparse import coo_matrix, random
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

# from etl import reviews_preprocessor
from etl.reviews_dataset_analyzer import ReviewsDatasetAnalyzer
from topicmodeling.context import topic_model_creator
# from etl.reviews_preprocessor import ReviewsPreprocessor
from topicmodeling.context.topic_model_analyzer import split_topic, \
    generate_excel_file
from topicmodeling.hungarian import HungarianError
from topicmodeling.jaccard_similarity import AverageJaccard
from topicmodeling.jaccard_similarity import RankingSetAgreement
from tripadvisor.fourcity import extractor
from utils import utilities
from utils.constants import Constants
from langdetect import detect
import operator
import random
__author__ = 'fpena'

# import sys
# sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')

# from topicmodeling.hiddenfactortopics.topic_corpus import TopicCorpus

# latent_reg = 0
# lambda_param = 0.1
# num_topics = 5
# # file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-short.votes'
# file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-shuffled.votes'
# # file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-short-shuffled.votes'
# corpus = Corpus()
# corpus.load_data(file_name, 0)
# topic_corpus = TopicCorpus(corpus, num_topics, latent_reg, lambda_param)
# topic_corpus.train(2, 50)

# my_file = '/Users/fpena/UCC/Thesis/external/McAuley2013/code_RecSys13/Arts.votes'
# my_corpus = Corpus(my_file, 0)
# my_corpus.process_reviews(my_file, 0)
# my_corpus.build_votes(my_file, 0)

import sys
sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')

import cPickle as pickle
import time
import numpy
from sklearn.cross_validation import KFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import tree
import matplotlib.pyplot as plt

from etl import ETLUtils
from topicmodeling.context import review_metrics_extractor
from topicmodeling.context.review import Review


def main():
    item_type = 'hotel'
    # item_type = 'restaurant'
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    my_file = my_folder + 'classified_' + item_type + '_reviews.json'
    binary_reviews_file = my_folder + 'classified_' + item_type + '_reviews.pkl'
    my_records = ETLUtils.load_json_file(my_file)

    with open(binary_reviews_file, 'rb') as read_file:
        my_reviews = pickle.load(read_file)

    num_features = 2

    my_metrics = numpy.zeros((len(my_reviews), num_features))
    for index in range(len(my_reviews)):
        my_metrics[index] =\
            review_metrics_extractor.get_review_metrics(my_reviews[index])

    review_metrics_extractor.normalize_matrix_by_columns(my_metrics)

    count_specific = 0
    count_generic = 0
    for record in my_records:

        if record['specific'] == 'yes':
            count_specific += 1

        if record['specific'] == 'no':
            count_generic += 1

    print('count_specific: %d' % count_specific)
    print('count_generic: %d' % count_generic)
    print('specific percentage: %f%%' % (float(count_specific)/len(my_records)))
    print('generic percentage: %f%%' % (float(count_generic)/len(my_records)))

    my_labels = numpy.array([record['specific'] == 'yes' for record in my_records])

    classifiers = [
        DummyClassifier(strategy='most_frequent', random_state=0),
        DummyClassifier(strategy='stratified', random_state=0),
        DummyClassifier(strategy='uniform', random_state=0),
        # DummyClassifier(strategy='constant', random_state=0, constant=True),
        LogisticRegression(C=100),
        SVC(C=1.0, kernel='rbf'),
        SVC(C=1.0, kernel='linear'),
        KNeighborsClassifier(n_neighbors=10),
        tree.DecisionTreeClassifier(),
        NuSVC(),
        LinearSVC()
    ]
    scores = [[] for _ in range(len(classifiers))]

    Xtrans = my_metrics

    cv = KFold(n=len(my_metrics), n_folds=5)

    for i in range(len(classifiers)):
        for train, test in cv:
            x_train, y_train = Xtrans[train], my_labels[train]
            x_test, y_test = Xtrans[test], my_labels[test]

            clf = classifiers[i]
            clf.fit(x_train, y_train)
            scores[i].append(clf.score(x_test, y_test))

    for classifier, score in zip(classifiers, scores):
        print("Mean(scores)=%.5f\tStddev(scores)=%.5f" % (numpy.mean(score), numpy.std(score)))

    plot(my_metrics, my_labels)


def plot(my_metrics, my_labels):
    clf = LogisticRegression(C=100)
    clf.fit(my_metrics, my_labels)

    def f_learned(lab):
        return clf.intercept_ + clf.coef_ * lab

    coef = clf.coef_[0]
    intercept = clf.intercept_

    # print('coef', coef)
    # print('intercept', intercept)

    xvals = numpy.linspace(0, 1.0, 2)
    yvals = -(coef[0] * xvals + intercept[0]) / coef[1]
    plt.plot(xvals, yvals, color='g')

    plt.xlabel("log number of words (normalized)")
    plt.ylabel("log number of verbs in past tense (normalized)")
    my_legends = ['Generic reviews', 'Specific reviews']
    for outcome, marker, colour, legend in zip([0, 1], "ox", "br", my_legends):
        plt.scatter(
            my_metrics[:, 0][my_labels == outcome],
            my_metrics[:, 1][my_labels == outcome], c = colour, marker = marker, label=legend)
    # plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
    plt.legend(loc='lower center', numpoints=1, ncol=3)
    # plt.legend(numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0, 1, 1))
    plt.show()

# main2()

#
# logging.config.dictConfig({
#     'version': 1,
#     'disable_existing_loggers': False,  # this fixes the problem
#
#     'formatters': {
#         'standard': {
#             'format': '%(asctime)s [%(levelname)s]\t %(name)s: %(message)s'
#             # 'format': '[%(levelname)s] %(message)s [%(module)s %(funcName)s %(lineno)d]'
#         },
#     },
#     'handlers': {
#         'default': {
#             'level':'DEBUG',
#             'class':'logging.StreamHandler',
#             'formatter': 'standard',
#         },
#     },
#     'loggers': {
#         '': {
#             'handlers': ['default'],
#             'level': 'DEBUG',
#             'propagate': True
#         }
#     }
# })
#
# logger = logging.getLogger(__name__)
#
# def test():
#     logger.debug('hola')

# logger.debug("Debug")
# logger.info("Total time = %f seconds" % total_time)


def main2():

    records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)[:10]
    # for record in records:
    #     print(record)

    cols = [
        Constants.USER_ID_FIELD,
        Constants.ITEM_ID_FIELD,
        Constants.RATING_FIELD
    ]
    data_frame = pandas.DataFrame(records, columns=cols)
    # print(data_frame)
    # data_frame['a'] = data_frame[Constants.USER_ID_FIELD].astype('category')
    # data_frame['b'] = data_frame[Constants.ITEM_ID_FIELD].astype('category')
    data_frame[Constants.USER_ID_FIELD] = data_frame[Constants.USER_ID_FIELD].astype('category')
    data_frame[Constants.ITEM_ID_FIELD] = data_frame[Constants.ITEM_ID_FIELD].astype('category')
    # category_columns = data_frame.select_dtypes(['category']).columns
    # print(category_columns)
    # data_frame[category_columns] = \
    #     data_frame[category_columns].apply(lambda x: x.cat.codes)
    # print(data_frame)
    # print(data_frame['b'].cat.categories[0])
    print(data_frame[Constants.USER_ID_FIELD].cat.codes)
    print(data_frame[Constants.ITEM_ID_FIELD].cat.codes)

    plays = coo_matrix((data_frame[Constants.RATING_FIELD].astype(float),
                        (data_frame[Constants.USER_ID_FIELD].cat.codes,
                         data_frame[Constants.ITEM_ID_FIELD].cat.codes)))

    print(plays)
    # from sklearn.decomposition import NMF
    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(plays)
    H = model.components_
    nR = numpy.dot(W, H)
    # print(nR)
    # print(nR.shape)

    print 'User-based CF MSE: ' + str(
        mean_squared_error(nR, plays.toarray()))

    # get SVD components from train matrix. Choose k.
    u, s, vt = svds(plays, k=5)
    s_diag_matrix = numpy.diag(s)
    X_pred = numpy.dot(numpy.dot(u, s_diag_matrix), vt)
    # print(X_pred)
    print 'User-based CF MSE: ' + str(mean_squared_error(X_pred, plays.toarray()))


def records_to_matrix(records, context_rich_topics=None):
    """
    Converts the given records into a numpy matrix, transforming given each
    string value an integer ID

    :param records: a list of dictionaries with the reviews information
    :type context_rich_topics list[(float, float)]
    :param context_rich_topics: a list that indicates which are the
    context-rich topics. The list contains pairs, in which the first position
    indicates the topic ID, and the second position, the ratio of the topic.
    :return: a numpy matrix with all the independent variables (X) and a numpy
    vector with all the dependent variables (y)
    """

    matrix = []
    y = []

    users_map = {}
    items_map = {}
    user_index = 0
    item_index = 0

    for record in records:
        user_id = record['user_id']
        item_id = record['business_id']
        y.append(record['stars'])

        if user_id not in users_map:
            users_map[user_id] = user_index
            user_index += 1

        if item_id not in items_map:
            items_map[item_id] = item_index
            item_index += 1

        row = [users_map[user_id], items_map[item_id]]

        # for topic in range(num_topics):
        #     topic_probability = record['lda_context'][topic]
        #     row.append(topic_probability)

        if Constants.USE_CONTEXT:
            for topic in context_rich_topics:
                # topic_probability = record['lda_context'][topic[0]]
                # row.append(topic_probability)
                # print(record)
                # print(record[Constants.CONTEXT_TOPICS_FIELD])
                topic_key = 'topic' + str(topic[0])
                topic_probability =\
                    record[Constants.CONTEXT_TOPICS_FIELD][topic_key]
                # print('topic_probability', topic_probability)
                row.append(topic_probability)
                # print('topic_probability', topic_probability)

        matrix.append(row)

    print('my num users: %d' % len(users_map))
    print('my num items: %d' % len(items_map))

    y = numpy.array(y)

    return matrix, y


def analyze_fourcity():
    records = ETLUtils.load_json_file(Constants.FULL_PROCESSED_RECORDS_FILE)
    # for record in records:
    #     print(record)

    cols = [
        Constants.USER_ID_FIELD,
        Constants.ITEM_ID_FIELD,
        Constants.RATING_FIELD
    ]
    data_frame = pandas.DataFrame(records, columns=cols)
    print(data_frame.describe())

    zero_records = 0
    for record in records:
        if record[Constants.RATING_FIELD] < 1.0:
            print(record)
            zero_records += 1

    for record in records:
        if record[Constants.RATING_FIELD] > 5.0:
            print(record)
            zero_records += 1
    print('zero records: %d' % zero_records)

    # Look for duplicates
    keys_set = set()
    num_duplicates = 0
    print('Looking for duplicates')
    records_map = {}
    for record in records:
        # if record[Constants.USER_ITEM_KEY_FIELD] in keys_set:
        record_key = record[Constants.USER_ITEM_KEY_FIELD]
        if record_key in records_map:
            print('old record', records_map[record_key][Constants.TEXT_FIELD])
            print('new record', record[Constants.TEXT_FIELD])
            num_duplicates += 1
        keys_set.add(record_key)
        records_map[record_key] = record

    print('duplicate records: %d' % num_duplicates)


def predict_matrix_factorization(self):

    if Constants.USE_CONTEXT:
        msg = 'Contextual information cannot be used with matrix ' \
              'factorization, please set the use_context variable to ' \
              'false or use another prediction method'
        raise ValueError(msg)

    all_records = self.train_records + self.records_to_predict

    # print('train records', self.train_records)
    # print('records to predict', self.records_to_predict)

    cols = [
        Constants.USER_ID_FIELD,
        Constants.ITEM_ID_FIELD,
        Constants.RATING_FIELD
    ]
    data_frame = pandas.DataFrame(all_records, columns=cols)
    data_frame[Constants.USER_ID_FIELD] = data_frame[
        Constants.USER_ID_FIELD].astype('category')
    data_frame[Constants.ITEM_ID_FIELD] = data_frame[
        Constants.ITEM_ID_FIELD].astype('category')

    full_matrix = coo_matrix((
        data_frame[Constants.RATING_FIELD].astype(float),
        (
            data_frame[Constants.USER_ID_FIELD].cat.codes,
            data_frame[Constants.ITEM_ID_FIELD].cat.codes
        )
    ))

    user_id_map = dict(zip(data_frame[Constants.USER_ID_FIELD], data_frame[Constants.USER_ID_FIELD].cat.codes))
    item_id_map = dict(zip(data_frame[Constants.ITEM_ID_FIELD], data_frame[Constants.ITEM_ID_FIELD].cat.codes))

    # print(len(data_frame))
    # print(len(user_id_map))
    # print(user_id_map)
    # print(user_id_map['XUuIkTfKf-lYf4JGhNRuHw'])

    # print(data_frame[Constants.USER_ID_FIELD].cat.categories['lYf4JGhNRuHw'])

    print('num users: %d' % len(set(data_frame[Constants.USER_ID_FIELD].cat.codes)))
    print('num items: %d' % len(set(data_frame[Constants.ITEM_ID_FIELD].cat.codes)))

    # print(full_matrix)
    # print(full_matrix.shape)
    # from sklearn.decomposition import NMF
    # model = NMF(n_components=2, init='random', random_state=0)
    # W = model.fit_transform(full_matrix)
    # H = model.components_
    # nR = numpy.dot(W, H)
    # print(nR)
    # print(nR.shape)

    # print 'User-based CF MSE: ' + str(
    #     mean_squared_error(nR, full_matrix.toarray()))

    train_matrix = full_matrix.toarray()[:len(self.train_records)]

    # get SVD components from train matrix. Choose k.
    # u, s, vt = svds(train_matrix, k=200)
    # s_diag_matrix = numpy.diag(s)
    # X_pred = numpy.dot(numpy.dot(u, s_diag_matrix), vt)
    # print(X_pred)
    # print(X_pred)
    # print(X_pred.shape)

    model = NMF(n_components=60, init='random', random_state=0)
    W = model.fit_transform(train_matrix)
    H = model.components_
    X_pred = numpy.dot(W, H)

    x_matrix, y_vector = records_to_matrix(
        all_records, self.context_rich_topics)

    print('matrix', len(x_matrix))
    print('full matrix len', len(full_matrix.toarray()))

    predictions = []
    for rating in self.records_to_predict:
        user_id = rating['user_id']
        item_id = rating['business_id']
        user_code = user_id_map[user_id]
        item_code = item_id_map[item_id]
        predictions.append(X_pred[user_code][item_code])
    # print('predictions', predictions)
    self.predictions = predictions
    #
    # x_train = x_matrix[:len(self.train_records)]
    # y_train = y_vector[:len(self.train_records)]
    # x_test = x_matrix[len(self.train_records):]


def build_manual_topic_model():

    new_classified_records_file = Constants.DATASET_FOLDER + 'classified_' + \
          Constants.ITEM_TYPE + '_reviews_first_sentences.json'
    records = ETLUtils.load_json_file(new_classified_records_file)
    # records = ETLUtils.filter_records(records, 'context_type', ['context'])
    # records = ETLUtils.filter_records(records, 'sentence_type', ['specific'])
    # records = ETLUtils.filter_records(records, 'sentence_index', [0])
    print('total records: %d' % len(records))

    # print(records[0])
    count = 0

    for i in range(len(records)):
        record = records[i]
        if record['sentence_index'] == 0.0:
            # if record['context_type'] == 'context' and record['context_summary'] != 'all_context':
            if record['sentence_type'] == 'specific':
                print('%d:\t%s' % (i+1, records[i]['text']))
                count += 1

    print('count: %d' % count)


def transform_manually_labeled_reviews():

    full_records = ETLUtils.load_json_file(Constants.DATASET_FOLDER + 'yelp_training_set_review_restaurants_shuffled_tagged.json')

    records = ETLUtils.load_json_file(Constants.CLASSIFIED_RECORDS_FILE)
    print('total records: %d' % len(records))

    new_records = []

    for record in records:

        sentence_index = record['sentence_index']

        if sentence_index > 0:
            continue
        record['predicted_class'] = record['sentence_type']
        new_records.append(record)

    # count = 0
    # for new_record in new_records:
    #     internal_count = 0
    #     for full_record in full_records:
    #         if full_record['text'].startswith(new_record['text']):
    #             # print(full_record['text'])
    #             internal_count += 1
    #             count += 1
    #             print('internal count: %d\treview_id: %s' % (internal_count, full_record['review_id']))
    #
    #             if internal_count > 1:
    #                 print('internal count: %d\treview_id: %s' % (internal_count, new_record['text']))

    # print('count: %d' % count)

    index = 0

    for new_record in new_records:

        while True:

            full_record = full_records[index]

            if full_record['text'].startswith(new_record['text']):
                new_record[Constants.USER_ID_FIELD] = full_record['user_id']
                new_record[Constants.ITEM_ID_FIELD] = full_record['business_id']
                new_record[Constants.REVIEW_ID_FIELD] = full_record['review_id']
                new_record[Constants.RATING_FIELD] = full_record['stars']
                break
            index += 1
        index += 1

    print('index: %d' % index)

    for new_record in new_records:

        for full_record in full_records:
            if new_record['review_id'] == full_record['review_id']:
                print('%s ====' % new_record['text'])
                print(full_record['text'])
                print('******************\n******************\n******************\n******************')
                break

    # reviews_preprocessor = ReviewsPreprocessor()
    # new_records = reviews_preprocessor.lemmatize_sentences(new_records)
    # reviews_preprocessor.records = new_records
    # reviews_preprocessor.build_bag_of_words()
    # reviews_preprocessor.drop_unnecessary_fields()

    new_classified_records_file = Constants.DATASET_FOLDER + 'classified_' + \
        Constants.ITEM_TYPE + '_reviews_first_sentences.json'

    print(new_records[0])

    ETLUtils.save_json_file(new_classified_records_file, new_records)

    # print('keys', records[0].keys())


# def preprocess_manually_labeled_reviews():
#
#     new_classified_records_file = Constants.DATASET_FOLDER + 'classified_' + \
#           Constants.ITEM_TYPE + '_reviews_first_sentences.json'
#     records = ETLUtils.load_json_file(new_classified_records_file)
#     print('total records: %d' % len(records))
#
#     reviews_preprocessor = ReviewsPreprocessor()
#     records = reviews_preprocessor.lemmatize_sentences(records)
#     reviews_preprocessor.records = records
#     reviews_preprocessor.build_bag_of_words()
#     # reviews_preprocessor.build_corpus()
#
#     for record in records:
#         record[Constants.PREDICTED_CLASS_FIELD] = record['sentence_type']
#
#     unnecessary_fields = [
#         'sentence_type',
#         'quartile_index',
#         'quintile_index',
#         'percentile',
#         'pos_tags',
#         'sentence_relative_index',
#         'is_review_specific',
#         'review_num_sentences',
#     ]
#
#     ETLUtils.drop_fields(unnecessary_fields, records)
#
#     print(records[0].keys())
#     print(records[0])
#
#     processed_records_file = Constants.generate_file_name(
#         'classified_processed_reviews', 'json', Constants.CACHE_FOLDER, None,
#         None, False, True)
#
#     ETLUtils.save_json_file(processed_records_file, records)


def create_topic_model_with_context_records():

    processed_records_file = Constants.generate_file_name(
        'classified_processed_reviews', 'json', Constants.CACHE_FOLDER, None,
        None, False, True)
    records = ETLUtils.load_json_file(processed_records_file)
    print('records length: %d' % len(records))

    context_records = ETLUtils.filter_records(records, 'context_type', ['context'])
    print('context records length: %d' % len(context_records))
    context_specific_records = ETLUtils.filter_records(context_records, 'predicted_class', ['specific'])
    print('context specific records length: %d' % len(context_specific_records))

    for i in range(len(context_specific_records)):
        # print('%d:\t%s' % (i, context_records[i]['text']))
        print('%d:\t%s' % (i, context_specific_records[i]['bow']))

    for i in range(1, len(context_records)+1):

        Constants.update_properties({Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: i})
        context_extractor = \
            topic_model_creator.create_topic_model(records, None, None)

        topic_data = []

        for topic in range(Constants.TOPIC_MODEL_NUM_TOPICS):
            result = {}
            result['topic_id'] = topic
            result.update(split_topic(context_extractor.print_topic_model(
                num_terms=Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)[topic]))
            result['ratio'] = context_extractor.topic_ratio_map[topic]
            result['weighted_frequency'] = \
                context_extractor.topic_weighted_frequency_map[topic]
            topic_data.append(result)

        file_name = Constants.generate_file_name(
            'manual_topic_model', 'xlsx', Constants.DATASET_FOLDER, None, None, True)
        generate_excel_file(topic_data, file_name)


def analyze_context_records():
    records = ETLUtils.load_json_file(Constants.CLASSIFIED_RECORDS_FILE)
    records = ETLUtils.filter_records(records, 'context_type', ['context'])

    print('num records: %d' % len(records))

    for record in records:
        print(record[Constants.TEXT_FIELD])


def generate_topic_model_creation_script():

    command = "PYTHONPATH='/home/fpena/yelp/source/python' stdbuf -oL nohup " \
              "python /home/fpena/yelp/source/python/topicmodeling/context/topic_model_creator.py "
    args = "-c 0 -f %d -t %d "
    suffix = "> ~/logs/topicmodel-hotel-%d-1-%d.log"

    full_command = command + args + suffix

    num_folds = range(10)
    num_topics_list = range(1, 51)

    for num_topics, fold in itertools.product(num_topics_list, num_folds):
        # print('fold', fold)
        # print('num_topics', num_topics)
        print(full_command % (fold, num_topics, num_topics, fold + 1))


def calculate_topic_stability(records):

    Constants.update_properties({
        Constants.NUMPY_RANDOM_SEED_FIELD: Constants.NUMPY_RANDOM_SEED + 10,
        Constants.RANDOM_SEED_FIELD: Constants.RANDOM_SEED + 10
    })
    utilities.plant_seeds()
    Constants.print_properties()

    if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
        num_records = len(records)
        records = records[:num_records / 2]
    print('num_reviews', len(records))

    all_term_rankings = []

    context_extractor =\
        topic_model_creator.create_topic_model(records, None, None)
    terms_matrix = get_topic_model_terms(
        context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
    all_term_rankings.append(terms_matrix)

    sample_ratio = 0.8

    print('Total iterations: %d' % Constants.TOPIC_MODEL_STABILITY_ITERATIONS)
    for _ in range(Constants.TOPIC_MODEL_STABILITY_ITERATIONS - 1):
        sampled_records = sample_list(records, sample_ratio)
        context_extractor = \
            topic_model_creator.train_context_extractor(sampled_records)
        terms_matrix = get_topic_model_terms(
            context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
        all_term_rankings.append(terms_matrix)

    return calculate_stability(all_term_rankings)


def calculate_stability(all_term_rankings):

    # First argument was the reference term ranking
    reference_term_ranking = all_term_rankings[0]
    all_term_rankings = all_term_rankings[1:]
    r = len(all_term_rankings)
    print("Loaded %d non-reference term rankings" % r)

    # Perform the evaluation
    metric = AverageJaccard()
    matcher = RankingSetAgreement(metric)
    print("Performing reference comparisons with %s ..." % str(metric))
    all_scores = []
    for i in range(r):
        try:
            score = \
                matcher.similarity(reference_term_ranking,
                                   all_term_rankings[i])
            all_scores.append(score)
        except HungarianError:
            msg = \
                "HungarianError: Unable to find results. Algorithm has failed."
            print(msg)
            all_scores.append(float('nan'))

    # Get overall score across all candidates
    all_scores = numpy.array(all_scores)

    print("Stability=%.4f [%.4f,%.4f] for %d topics" % (
        numpy.nanmean(all_scores), numpy.nanmin(all_scores),
        numpy.nanmax(all_scores), Constants.TOPIC_MODEL_NUM_TOPICS))

    return all_scores


def get_topic_model_terms(context_extractor, num_terms):

    context_extractor.num_topics = Constants.TOPIC_MODEL_NUM_TOPICS
    topic_model_strings = context_extractor.print_topic_model(num_terms)
    topic_term_matrix = []

    for topic in range(Constants.TOPIC_MODEL_NUM_TOPICS):
        terms = topic_model_strings[topic].split(" + ")
        terms = [term.partition("*")[2] for term in terms]
        topic_term_matrix.append(terms)

    return topic_term_matrix


def sample_list(lst, sample_ratio):

    num_samples = int(len(lst) * sample_ratio)
    sampled_list = [
        lst[i] for i in sorted(random.sample(xrange(len(lst)), num_samples))]

    return sampled_list


def topic_stability_main():

    records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)

    # num_topic_list = range(2, 101)
    num_topic_list = [2, 5]
    results = {}
    for num_topics in num_topic_list:
        new_properties = {Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: num_topics}
        Constants.update_properties(new_properties)
        results[num_topics] = calculate_topic_stability(records)

    print('Results:')
    for num_topics in num_topic_list:
        scores = results[num_topics]
        print('%d: %.4f [%.4f,%.4f]' %
              (num_topics, numpy.nanmean(scores), numpy.nanmin(scores),
               numpy.nanmax(scores)))


def factorize_nmf():
    print('factorizing matrix')

    newsgroups_mmf_file = '/Users/fpena/tmp/nmf_graphlab/newsgroups/newsgroups_matrix.mmf'
    document_term_matrix = mmread(newsgroups_mmf_file)

    factorizer = decomposition.NMF(
        init="nndsvd", n_components=Constants.TOPIC_MODEL_NUM_TOPICS,
        max_iter=Constants.TOPIC_MODEL_ITERATIONS,
        alpha=Constants.NMF_REGULARIZATION,
        l1_ratio=Constants.NMF_REGULARIZATION_RATIO
    )
    document_topic_matrix = \
        factorizer.fit_transform(document_term_matrix)
    topic_term_matrix = factorizer.components_
    # mmwrite(mmf_file, small_matrix)
    # mmwrite(newsgroups_mmf_file, X)


def export_records_to_text():
    print('%s: Exporting bag-of-words to text files' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))

    records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)
    print('Total records: %d' % len(records))

    folder = '/Users/fpena/tmp/topic-ensemble/data/' + Constants.ITEM_TYPE + '/'

    for record in records:
        file_name = folder + Constants.ITEM_TYPE + '_' + \
                    record[Constants.REVIEW_ID_FIELD] + '.txt'

        with codecs.open(file_name, "w", encoding="utf-8-sig") as text_file:
            text_file.write(" ".join(record[Constants.BOW_FIELD]))


def dataset_bucket_analysis_by_field(field):
    # Set the dataset
    hotel_dataset_properties = {Constants.BUSINESS_TYPE_FIELD: 'fourcity_hotel'}
    Constants.update_properties(hotel_dataset_properties)

    records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)

    print('Loaded %d records' % len(records))

    user_frequency_map = {}

    for record in records:

        user_id = record[field]
        if user_id not in user_frequency_map:
            user_frequency_map[user_id] = 0
        user_frequency_map[user_id] += 1

    print('There is a total of %d %ss' % (len(user_frequency_map), field))
    sorted_x = sorted(user_frequency_map.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_x[0])
    print(sorted_x[1])
    print(sorted_x[2])
    # print(user_frequency_map)

    # Number of reviews per user
    rda = ReviewsDatasetAnalyzer(records)
    users_summary = rda.summarize_reviews_by_field(field)
    print('Average number of reviews per %s: %f' % (field,
          float(rda.num_reviews) / rda.num_users))
    users_summary.plot(kind='line', rot=0)

    pandas.set_option('display.max_rows', len(users_summary))
    print(users_summary)
    pandas.reset_option('display.max_rows')

    # print(users_summary)
    # plt.show()


def remove_duplicate_records(records, field):
    print('%s: remove duplicate records' % time.strftime(
        "%Y/%m/%d-%H:%M:%S"))

    ids_set = set()
    non_duplicated_records = []

    for record in records:
        if record[field] not in ids_set:
            ids_set.add(record[field])
            non_duplicated_records.append(record)

    return non_duplicated_records


def parse_dafevara_file():

    ARTISTS_NAMES_FIELD = 'artists_names'
    folder = '/Users/fpena/tmp/dafevara/'
    file_path = folder + 'artists-names-by-userId.csv'
    records = ETLUtils.load_csv_file(file_path, '|')

    for record in records:
        artists = record[ARTISTS_NAMES_FIELD].replace(' ', '_')
        record[ARTISTS_NAMES_FIELD] = artists.replace(';', ' ')
        # print(record[ARTISTS_NAMES_FIELD])

    output_file = folder + 'user_artists.txt'
    with open(output_file, 'w') as of:
        for record in records:
            of.write('%s\n' % record[ARTISTS_NAMES_FIELD])


def generate_commands():
    command = "PYTHONPATH='/home/fpena/yelp/source/python' stdbuf -oL nohup python " \
              "/home/fpena/yelp/source/python/main/recommender_runner.py -k %d -i %s > " \
              "/home/fpena/logs/%s_recommender_runner_ctw%d_ensemble.log"

    for i in [12, 13, 24]:
        print(command % (i, Constants.ITEM_TYPE, Constants.ITEM_TYPE, i))


def generate_commands_java():
    command = "stdbuf -oL nohup java -jar " \
              "/home/fpena/yelp/source/java/richcontext/target/richcontext-1.0-SNAPSHOT-jar-with-dependencies.jar" \
              " -p /home/fpena/yelp/source/python/properties.yaml" \
              " -d /home/fpena/data/cache_context/" \
              " -o /home/fpena/data/results/ -t process_libfm_results" \
              " -s test_only_users -k %d -i %s" \
              " > /home/fpena/logs/%s_evaluate_ctw%d_ensemble_cold_start.log"

    for i in range(2, 61):
        print(command % (i, Constants.ITEM_TYPE, Constants.ITEM_TYPE, i))


def add_extra_column_to_csv():

    csv_file_name = '/tmp/results/rival_yelp_restaurant_results_folds_4.csv'

    records = ETLUtils.load_csv_file(csv_file_name)

    with open(csv_file_name, 'r') as csvinput:
        reader = csv.reader(csvinput)
        headers = next(reader)
        index = headers.index('Evaluation_Set') + 1
        headers.insert(index, Constants.FM_NUM_FACTORS_FIELD)

    print(headers)

    for record in records:
        record[Constants.FM_NUM_FACTORS_FIELD] = 10

    ETLUtils.save_csv_file('/tmp/my_csv_file.csv', records, headers)

    # print(records)


start = time.time()
# parse_dafevara_file()
# analyze_context_records()
# generate_topic_model_creation_script()
# topic_stability_main()
# factorize_nmf()
# export_records_to_text()
# analyze_fourcity()
# dataset_bucket_analysis_by_field(Constants.ITEM_ID_FIELD)
# generate_commands()
generate_commands_java()
# add_extra_column_tocsv()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)


# print('file: %s' % Constants.CLASSIFIED_RECORDS_FILE)
# records = ETLUtils.load_json_file(Constants.CLASSIFIED_RECORDS_FILE)
#
# count_map = {}
# for record in records:
#     if record['sentence_type'] == 'generic':
#         print('\n\n\n*************************\n%s' % record['text'])
#

# x = [0.308383458, 0.315857778, 0.301238078, 0.309317792, 0.303709392, 0.40291414]
# y = [0.310611881, 0.317933334, 0.306328969, 0.312251411, 0.30976714, 0.405549174]
#
# rmse1 = [0.40291414, 0.411198607, 0.397752339, 0.405695217, 0.400702497]
# rmse2 = [0.405549174, 0.413125616, 0.401288785, 0.408203643, 0.404572212]
#
# print(wilcoxon(x, y, zero_method='wilcox', correction=False))
# print (ttest_ind(x, y, equal_var=False))
# print (ttest_ind(rmse1, rmse2, equal_var=False))
# print(wilcoxon(rmse1, rmse2, zero_method='wilcox', correction=False))


# dataset_bucket_analysis()

