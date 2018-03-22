import random

import numpy
import numbers
import time

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import tree

from etl import ETLUtils
from etl.reviews_preprocessor import ReviewsPreprocessor
from topicmodeling.context import review_metrics_extractor
from utils.constants import Constants


RANDOM_STATE = 0
SCORE_METRIC = 'accuracy'
# SCORE_METRIC = 'roc_auc'
resamplers = [
    None,
    RandomUnderSampler(random_state=RANDOM_STATE),
    TomekLinks(random_state=RANDOM_STATE),
    EditedNearestNeighbours(random_state=RANDOM_STATE),
    NeighbourhoodCleaningRule(random_state=RANDOM_STATE),
    RandomOverSampler(random_state=RANDOM_STATE),
    SMOTE(random_state=RANDOM_STATE),
    SMOTETomek(random_state=RANDOM_STATE),
    SMOTEENN(random_state=RANDOM_STATE)
]


PARAM_GRID_MAP = {
    'DummyClassifier': {
        'resampler': resamplers,
        'classifier': [DummyClassifier(random_state=RANDOM_STATE)],
        'classifier__strategy': ['most_frequent', 'stratified', 'uniform']
    },
    'LogisticRegression': {
        'resampler': resamplers,
        'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
        'classifier__C': [0.1, 1.0, 10, 100, 1000]
        # 'classifier__C': [0.1, 1.0, 10]
    },
    'SVC': {
        'resampler': resamplers,
        'classifier': [SVC(random_state=RANDOM_STATE)],
        'classifier__kernel': ['rbf', 'linear'],
        'classifier__C': [0.1, 1.0, 10, 100, 1000]
        # 'classifier__C': [0.1, 1.0, 10]
    },
    'KNeighborsClassifier': {
        'resampler': resamplers,
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [1, 2, 5, 10, 20],
        'classifier__weights': ['uniform', 'distance']
    },
    'DecisionTreeClassifier': {
        'resampler': resamplers,
        'classifier': [tree.DecisionTreeClassifier(random_state=RANDOM_STATE)],
        'classifier__max_depth': [None, 2, 3, 5, 10],
        'classifier__min_samples_leaf': [2, 5, 10]
    },
    'RandomForestClassifier': {
        'resampler': resamplers,
        'classifier': [RandomForestClassifier(random_state=RANDOM_STATE)],
        'classifier__n_estimators': [10, 50, 100, 200]
    }
}


# PARAM_GRID_MAP = {
#     'SVC None': {
#         'resampler': [None],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'SVC.1 ' + type(resamplers[1]).__name__: {
#         'resampler': [resamplers[1]],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'SVC.2 ' + type(resamplers[2]).__name__: {
#         'resampler': [resamplers[2]],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'SVC.3 ' + type(resamplers[3]).__name__: {
#         'resampler': [resamplers[3]],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'SVC.4 ' + type(resamplers[4]).__name__: {
#         'resampler': [resamplers[4]],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'SVC.5 ' + type(resamplers[5]).__name__: {
#         'resampler': [resamplers[5]],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'SVC.6 ' + type(resamplers[6]).__name__: {
#         'resampler': [resamplers[6]],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'SVC.7 ' + type(resamplers[7]).__name__: {
#         'resampler': [resamplers[7]],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'SVC.8 ' + type(resamplers[8]).__name__: {
#         'resampler': [resamplers[8]],
#         'classifier': [SVC(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
# }


# PARAM_GRID_MAP = {
#     'LogisticRegression None': {
#         'resampler': [None],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'LogisticRegression.1 ' + type(resamplers[1]).__name__: {
#         'resampler': [resamplers[1]],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'LogisticRegression.2 ' + type(resamplers[2]).__name__: {
#         'resampler': [resamplers[2]],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'LogisticRegression.3 ' + type(resamplers[3]).__name__: {
#         'resampler': [resamplers[3]],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'LogisticRegression.4 ' + type(resamplers[4]).__name__: {
#         'resampler': [resamplers[4]],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'LogisticRegression.5 ' + type(resamplers[5]).__name__: {
#         'resampler': [resamplers[5]],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'LogisticRegression.6 ' + type(resamplers[6]).__name__: {
#         'resampler': [resamplers[6]],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'LogisticRegression.7 ' + type(resamplers[7]).__name__: {
#         'resampler': [resamplers[7]],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
#     'LogisticRegression.8 ' + type(resamplers[8]).__name__: {
#         'resampler': [resamplers[8]],
#         'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
#         'classifier__C': [0.1, 1.0, 10, 100, 1000]
#     },
# }


def load_records():
    """
    Loads the reviews that have been manually tagged at a sentence level,
    this are the reviews that we will use to train our classifier. Only the
    first sentence of each review will be used
    """

    print('%s: load records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    # file_name = '/Users/fpena/UCC/Thesis/datasets/context/oldClassifiedFiles/classified_yelp_hotel_reviews.json'
    # file_name = '/Users/fpena/UCC/Thesis/datasets/context/oldClassifiedFiles/classified_yelp_restaurant_reviews.json'
    # records = ETLUtils.load_json_file(file_name)
    records = ETLUtils.load_json_file(Constants.CLASSIFIED_RECORDS_FILE)

    # Take only the first sentence
    # document_level = 1
    if isinstance(Constants.DOCUMENT_LEVEL, (int, long)):
        records = [
            record for record in records
            if record['sentence_index'] < Constants.DOCUMENT_LEVEL
        ]

    return records


def transform(records):
    """
    Transforms the reviews into a numpy matrix so that they can be easily
    processed by the functions available in scikit-learn

    :type records: list[dict]
    :param records: a list of dictionaries with the reviews
    :return:
    """

    num_features =\
        len(review_metrics_extractor.get_review_metrics(records[0]))
    x_matrix = numpy.zeros((len(records), num_features))

    for index in range(len(records)):
        x_matrix[index] =\
            review_metrics_extractor.get_review_metrics(records[index])

    min_values = x_matrix.min(axis=0)
    max_values = x_matrix.max(axis=0)
    review_metrics_extractor.normalize_matrix_by_columns(
        x_matrix, min_values, max_values)

    y_vector =\
        numpy.array([record['specific'] == 'yes' for record in records])

    return x_matrix, y_vector


def preprocess_records(records):

    print('length before: %d' % len(records))

    empty_records = []
    if isinstance(Constants.DOCUMENT_LEVEL, (int, long)):
        for record in records:
            sentence_type = record['sentence_type']
            record['specific'] = \
                'yes' if sentence_type in ['specific', 'unknown'] else 'no'
            if sentence_type == 'empty':
                empty_records.append(record)

    for empty_record in empty_records:
        records.remove(empty_record)

    print('length after: %d' % len(records))

    # lemmatize_reviews(records)
    ReviewsPreprocessor.lemmatize_reviews(records)


def count_specific_generic(records):
    """
    Prints the proportion of specific and generic documents

    :type records: list[dict]
    :param records: a list of dictionaries with the reviews
    """
    count_specific = 0
    count_generic = 0
    for record in records:

        if record['specific'] == 'yes':
            count_specific += 1

        if record['specific'] == 'no':
            count_generic += 1

    specific_percentage = float(count_specific)/len(records)*100
    generic_percentage = float(count_generic)/len(records)*100

    print('Dataset: %s' % Constants.ITEM_TYPE)
    print('count_specific: %d' % count_specific)
    print('count_generic: %d' % count_generic)
    print('specific percentage: %f%%' % specific_percentage)
    print('generic percentage: %f%%' % generic_percentage)


def plant_random_seeds():
    numpy.random.seed(0)
    random.seed(0)


def full_cycle():

    plant_random_seeds()
    my_records = load_records()
    preprocess_records(my_records)
    x_matrix, y_vector = transform(my_records)
    count_specific_generic(my_records)

    # Error estimation
    best_classifier = None
    best_score = 0.0
    for classifier, params in PARAM_GRID_MAP.items():
        # print('Classifier: %s' % classifier)
        cv = StratifiedKFold(Constants.CROSS_VALIDATION_NUM_FOLDS)
        score = error_estimation(x_matrix, y_vector, params, cv, SCORE_METRIC).mean()
        print('%s score: %f' % (classifier, score))

        if score > best_score:
            best_score = score
            best_classifier = classifier

    # Model selection
    cv = StratifiedKFold(Constants.CROSS_VALIDATION_NUM_FOLDS)
    grid_search_cv = model_selection(
        x_matrix, y_vector, PARAM_GRID_MAP[best_classifier], cv, SCORE_METRIC)
    # best_model = grid_search_cv.best_estimator_.get_params()['classifier']
    # features_importance = best_model.coef_
    print('%s: %f' % (SCORE_METRIC, grid_search_cv.best_score_))
    print('best params', grid_search_cv.best_params_)
    print('best estimator', grid_search_cv.best_estimator_)
    # print('features importance', features_importance)

    # results = get_scores(final_grid_search_cv.cv_results_)
    # csv_file = '/Users/fpena/tmp/' + Constants.ITEM_TYPE + '_new_reviews_classifier_results.csv'
    # ETLUtils.save_csv_file(csv_file, results, results[0].keys())
    #
    # print(csv_file)


def error_estimation(
        x_matrix, y_vector, param_grid, cv=None, scoring=None):
    pipeline = Pipeline([('resampler', None), ('classifier', DummyClassifier())])
    grid_search_cv = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring)

    return cross_val_score(grid_search_cv, x_matrix, y_vector)


def model_selection(
        x_matrix, y_vector, param_grid, cv=None, scoring=None):
    pipeline = Pipeline(
        [('resampler', None), ('classifier', DummyClassifier())])
    grid_search_cv = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring)
    grid_search_cv.fit(x_matrix, y_vector)
    return grid_search_cv


def get_features_importance(estimator):

    features_importance = estimator.coef_
    print(features_importance)
    return features_importance


def get_param_value_name(param_value):

    if isinstance(param_value, basestring) or \
            isinstance(param_value, numbers.Number):
        return param_value

    return type(param_value).__name__

# start = time.time()
# full_cycle()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
