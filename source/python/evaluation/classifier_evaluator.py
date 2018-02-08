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

PARAM_GRID = [
    {
        'resampler': resamplers,
        'classifier': [DummyClassifier(random_state=RANDOM_STATE)],
        'classifier__strategy': ['most_frequent', 'stratified', 'uniform']
    },
    {
        'resampler': resamplers,
        'classifier': [
            LogisticRegression(random_state=RANDOM_STATE),
            SVC(random_state=RANDOM_STATE)
        ],
        'classifier__C': [0.1, 1.0, 10, 100, 1000]
        # 'classifier__C': [0.1, 1.0, 10]
    },
    {
        'resampler': resamplers,
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [1, 2, 5, 10, 20],
        'classifier__weights': ['uniform', 'distance']
    },
    {
        'resampler': resamplers,
        'classifier': [
            tree.DecisionTreeClassifier(random_state=RANDOM_STATE)
        ],
        'classifier__max_depth': [None, 2, 3, 5, 10],
        'classifier__min_samples_leaf': [2, 5, 10]
    },
    {
        'resampler': resamplers,
        'classifier': [RandomForestClassifier(random_state=RANDOM_STATE)],
        'classifier__n_estimators': [10, 50, 100, 200]
    }
]


def load_records():
    """
    Loads the reviews that have been manually tagged at a sentence level,
    this are the reviews that we will use to train our classifier. Only the
    first sentence of each review will be used
    """

    print('%s: load records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
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


def nested_grid_search_cross_validation():

    plant_random_seeds()
    my_records = load_records()
    preprocess_records(my_records)
    x_matrix, y_vector = transform(my_records)
    count_specific_generic(my_records)

    grid_list = create_grid_list()

    print('score metric: %s' % SCORE_METRIC)

    tuned_estimators_param_grid = {'classifier': grid_list}
    tuned_pipeline = Pipeline([('classifier', DummyClassifier())])
    outer_cv = StratifiedKFold(n_splits=5)
    tuned_grid_search_cv = GridSearchCV(
        tuned_pipeline, tuned_estimators_param_grid, cv=outer_cv,
        scoring=SCORE_METRIC)
    tuned_grid_search_cv.fit(x_matrix, y_vector)

    print('\nBest estimator')
    print(tuned_grid_search_cv.best_estimator_.get_params()['classifier'].best_estimator_)
    print('End best estimator\n')
    # print(tuned_grid_search_cv.best_params_)
    print('Best score: %f' % tuned_grid_search_cv.best_score_)
    print('Best classifier algorithm index: %d' % tuned_grid_search_cv.best_index_)

    print('\n\n***************')
    get_scores_deep(tuned_grid_search_cv.cv_results_)
    print('***************\n\n')

    final_cv = StratifiedKFold(n_splits=5)
    final_estimator = tuned_grid_search_cv.best_estimator_.get_params()['classifier'].best_estimator_
    final_grid_search_cv = GridSearchCV(
        final_estimator, PARAM_GRID[tuned_grid_search_cv.best_index_],
        cv=final_cv, scoring=SCORE_METRIC
    )
    final_grid_search_cv.fit(x_matrix, y_vector)
    print('\nFinal estimator')
    print(final_grid_search_cv.best_params_)
    print('End final estimator\n')
    print('Final score: %f' % final_grid_search_cv.best_score_)

    results = get_scores(final_grid_search_cv.cv_results_)
    csv_file = '/Users/fpena/tmp/' + Constants.ITEM_TYPE + '_new_reviews_classifier_results.csv'
    ETLUtils.save_csv_file(csv_file, results, results[0].keys())

    print(csv_file)


def grid_search_cross_validation():

    plant_random_seeds()
    my_records = load_records()
    preprocess_records(my_records)
    x_matrix, y_vector = transform(my_records)

    grid_list = create_grid_list()

    for grid in grid_list:
        grid.fit(x_matrix, y_vector)
        print('\nBest estimator')
        print(grid.best_estimator_.get_params()['classifier'])
        print('End best estimator\n')
        # print(tuned_grid_search_cv.best_params_)
        print('Best score: %f' % grid.best_score_)
        print('Best classifier algorithm index: %d\n' % grid.best_index_)


def create_grid_list():
    grid_list = [GridSearchCV(
        Pipeline([('resampler', None), ('classifier', DummyClassifier())]),
        params, cv=StratifiedKFold(n_splits=5), scoring=SCORE_METRIC)
                 for params in PARAM_GRID]
    return grid_list


def get_scores(cv_results):
    params = cv_results['params']
    scores = cv_results['mean_test_score']
    param_keys = params[0].keys()
    param_values_list = []
    for param, score in zip(params, scores):
        param_values = {'score': score}
        for param_key in param_keys:
            param_values[param_key] = get_param_value_name(param[param_key])

        param_values_list.append(param_values)

    for param_values in param_values_list:
        print(param_values)

    return param_values_list


def get_scores_deep(cv_results):
    params = cv_results['params']
    scores = cv_results['mean_test_score']
    param_keys = params[0].keys()
    param_values_list = []
    for param, score in zip(params, scores):
        param_values = {'score': score}
        for param_key in param_keys:
            param_values[param_key] = param[param_key].best_estimator_.get_params()['classifier']
            # param_values['one'] = param[param_key]
            # param_values['two'] = param[param_key].best_estimator_
            # param_values['three'] = param[param_key].best_estimator_.get_params()['classifier']

        param_values_list.append(param_values)

    for param_values in param_values_list:
        print(param_values)

    return param_values_list


def get_param_value_name(param_value):

    if isinstance(param_value, basestring) or \
            isinstance(param_value, numbers.Number):
        return param_value

    return type(param_value).__name__

start = time.time()
# main()
nested_grid_search_cross_validation()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
