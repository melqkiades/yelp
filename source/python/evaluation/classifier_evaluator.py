import json
import random

import numpy
import numbers
import os
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
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import tree

from etl import ETLUtils
from etl import sampler_factory
from nlp import nlp_utils
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
    # 'DummyClassifier': {
    #     'resampler': resamplers,
    #     'classifier': [DummyClassifier(random_state=RANDOM_STATE)],
    #     'classifier__strategy': ['most_frequent', 'stratified', 'uniform']
    # },
    'LogisticRegression': {
        'resampler': resamplers,
        'classifier': [LogisticRegression(random_state=RANDOM_STATE)],
        'classifier__C': [0.1, 1.0, 10, 100, 1000]
        # 'classifier__C': [0.1, 1.0, 10]
    },
    # 'SVC': {
    #     'resampler': resamplers,
    #     'classifier': [SVC(random_state=RANDOM_STATE)],
    #     'classifier__kernel': ['rbf', 'linear'],
    #     'classifier__C': [0.1, 1.0, 10, 100, 1000]
    #     'classifier__C': [0.1, 1.0, 10]
    # },
    # 'KNeighborsClassifier': {
    #     'resampler': resamplers,
    #     'classifier': [KNeighborsClassifier()],
    #     'classifier__n_neighbors': [1, 2, 5, 10, 20],
    #     'classifier__weights': ['uniform', 'distance']
    # },
    # 'DecisionTreeClassifier': {
    #     'resampler': resamplers,
    #     'classifier': [tree.DecisionTreeClassifier(random_state=RANDOM_STATE)],
    #     'classifier__max_depth': [None, 2, 3, 5, 10],
    #     'classifier__min_samples_leaf': [2, 5, 10]
    # },
    # 'RandomForestClassifier': {
    #     'resampler': resamplers,
    #     'classifier': [RandomForestClassifier(random_state=RANDOM_STATE)],
    #     'classifier__n_estimators': [10, 50, 100, 200]
    # }
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

    lemmatize_reviews(records)
    # ReviewsPreprocessor.lemmatize_reviews(records)


def lemmatize_reviews(records):
    """
    Performs a POS tagging on the text contained in the reviews and
    additionally finds the lemma of each word in the review

    :type records: list[dict]
    :param records: a list of dictionaries with the reviews
    """
    print('%s: lemmatize reviews' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    record_index = 0
    for record in records:
        #

        tagged_words =\
            nlp_utils.lemmatize_text(record[Constants.TEXT_FIELD])

        record[Constants.POS_TAGS_FIELD] = tagged_words
        record_index += 1

    return records


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

    # for key, value in grid_search_cv.best_params_.items():
    #     print(key, value)

    # print('best estimator', grid_search_cv.best_estimator_)
    # print('features importance', features_importance)

    # csv_file_name = Constants.generate_file_name(
    #     'classifier_results', 'csv', Constants.RESULTS_FOLDER, None,
    #     None, False)
    # json_file_name = Constants.generate_file_name(
    #     'classifier_results', 'json', Constants.RESULTS_FOLDER, None,
    #     None, False)

    # results = get_scores(final_grid_search_cv.cv_results_)
    # csv_file = '/Users/fpena/tmp/' + Constants.ITEM_TYPE + '_new_reviews_classifier_results.csv'
    # ETLUtils.save_csv_file(csv_file, results, results[0].keys())
    #
    # print(csv_file)

    best_hyperparams_file_name = Constants.generate_file_name(
        'best_hyperparameters', 'json', '/tmp/', None,
        None, False)
    save_parameters(best_hyperparams_file_name, grid_search_cv.best_params_)


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


def save_parameters(file_name, parameters):

    for key, value in parameters.items():

        if key in ['classifier', 'resampler']:
            # new_value = None
            if value is not None:
                value = type(value).__name__
            parameters[key] = value

        parameters[key] = value
        # print(key, parameters[key])

    file_contents = json.dumps(parameters)
    print(file_contents)
    with open(file_name, 'w') as json_file:
        json_file.write(file_contents)


def load_pipeline():

    best_hyperparams_file_name = Constants.generate_file_name(
        'best_hyperparameters', 'json', Constants.CACHE_FOLDER, None,
        None, False)

    if not os.path.exists(best_hyperparams_file_name):
        print('Recsys contextual records have already been generated')
        full_cycle()

    with open(best_hyperparams_file_name, 'r') as json_file:
        file_contents = json_file.read()
        parameters = json.loads(file_contents)

        print(parameters)

        classifiers = {
            'logisticregression': LogisticRegression(),
            'svc': SVC(),
            'kneighborsclassifier': KNeighborsClassifier(),
            'decisiontreeclassifier': DecisionTreeClassifier(),
            'nusvc': NuSVC(),
            'randomforestclassifier': RandomForestClassifier()
        }

        classifier = classifiers[parameters['classifier'].lower()]
        # print(classifier)
        classifier_params = get_classifier_params(parameters)
        classifier.set_params(**classifier_params)
        print(classifier)

        resampler = sampler_factory.create_sampler(
            parameters['resampler'], Constants.DOCUMENT_CLASSIFIER_SEED)

        return Pipeline([('resampler', resampler), ('classifier', classifier)])

        # for pname, pval in parameters.items():
        #     print(pname, pval)
        #
        #     if pname.startswith('classifier__'):
        #         step, param = pname.split('__', 1)
        #         # fit_params_steps[step][param] = pval
        #         print(step, param)


def get_classifier_params(parameters):

    classifier_params = {}

    for pname, pval in parameters.items():
        print(pname, pval)

        if pname.startswith('classifier__'):
            step, param = pname.split('__', 1)
            # fit_params_steps[step][param] = pval
            print(step, param)
            classifier_params[param] = pval

    # print(classifier_params)
    return classifier_params


def main():

    print(load_pipeline())


# start = time.time()
# full_cycle()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
