
import time

from sklearn.dummy import DummyClassifier
import sklearn.metrics as skmetrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree
from sklearn.svm import NuSVC
from topicmodeling.context import topic_model_creator

from sklearn.svm import SVC

from unbalanced_dataset.over_sampling import RandomOverSampler
from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.combine import SMOTETomek
from unbalanced_dataset.combine import SMOTEENN

from etl import ETLUtils
from etl.yelp_reviews_preprocessor import YelpReviewsPreprocessor
from utils.constants import Constants
from sklearn.cross_validation import StratifiedKFold
from nlp import nlp_utils
from topicmodeling.context import review_metrics_extractor
from sklearn.linear_model import LogisticRegression
import itertools
import numpy
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()


def load_records():
    """
    Loads the reviews that have been manually tagged at a sentence level,
    this are the reviews that we will use to train our classifier. Only the
    first sentence of each review will be used
    """

    print('%s: load records' % time.strftime("%Y/%m/%d-%H:%M:%S"))
    dataset = Constants.ITEM_TYPE
    folder = Constants.DATASET_FOLDER
    records_file = folder + \
                   'classified_' + dataset + '_reviews_sentences.json'
    records = ETLUtils.load_json_file(records_file)

    # Take only the first sentence
    # max_sentences = 1
    if Constants.MAX_SENTENCES is not None:
        records = [
            record for record in records
            if record['sentence_index'] < Constants.MAX_SENTENCES
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


def resample(x_matrix, y_vector, sampler_type):
    """
    Resamples a dataset with imbalanced data so that the labels contained in
    the y_vector are distributed equally. This is done to prevent a classifier
    from being biased by the number of sample of a certain class.

    :param x_matrix: a numpy matrix with the independent variables
    :param y_vector: a numpy vector with the dependent variables
    :param sampler_type: the type of sampler that is going to be used to
    resample the data
    :return: a numpy matrix and a numpy vector with the data resampled using the
    selected sampler
    """

    if sampler_type is None:
        return x_matrix, y_vector

    verbose = False
    ratio = 'auto'
    random_state = 0
    samplers = {
        'random_over_sampler': RandomOverSampler(
            ratio=ratio, verbose=verbose),
        'smote_regular': SMOTE(
            ratio=ratio, random_state=random_state, verbose=verbose,
            kind='regular'),
        'smote_bl1': SMOTE(
            ratio=ratio, random_state=random_state, verbose=verbose,
            kind='borderline1'),
        'smote_bl2': SMOTE(
            ratio=ratio, random_state=random_state, verbose=verbose,
            kind='borderline2'),
        'smote_tomek': SMOTETomek(
            ratio=ratio, random_state=random_state, verbose=verbose),
        'smoteenn': SMOTEENN(
            ratio=ratio, random_state=random_state, verbose=verbose)
    }

    sampler = samplers[sampler_type]
    resampled_x, resampled_y = sampler.fit_transform(x_matrix, y_vector)

    return resampled_x, resampled_y


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

# count_specific_generic()


def plot(records):
    num_features = len(review_metrics_extractor.get_review_metrics(records[0]))
    metrics = numpy.zeros((len(records), num_features))
    for index in range(len(records)):
        metrics[index] = \
            review_metrics_extractor.get_review_metrics(records[index])

    review_metrics_extractor.normalize_matrix_by_columns(metrics)
    labels = numpy.array([record['specific'] == 'yes' for record in records])

    clf = LogisticRegression(C=100)
    clf.fit(metrics, labels)

    coef = clf.coef_[0]
    intercept = clf.intercept_

    print('coef', coef)
    # print('intercept', intercept)

    xvals = numpy.linspace(0, 1.0, 2)
    yvals = -(coef[0] * xvals + intercept[0]) / coef[1]
    plt.plot(xvals, yvals, color='g', label='Decision boundary')

    plt.xlabel("log number of words (normalized)")
    plt.ylabel("log number of verbs in past tense (normalized)")
    my_legends = ['Specific reviews', 'Generic reviews']
    for outcome, marker, colour, legend in zip([0, 1], "ox", "br", my_legends):
        plt.scatter(
            metrics[:, 0][labels == outcome],
            metrics[:, 1][labels == outcome], c=colour, marker=marker,
            label=legend)
    # plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
    plt.legend(loc='lower left', numpoints=1, ncol=3, fontsize=8,
               bbox_to_anchor=(0, 0))
    # plt.savefig('/Users/fpena/tmp/restaurant_sentence_classification.pdf')


# plot()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    target_names = ['generic', 'specific']
    tick_marks = numpy.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def preprocess_records(records):

    print('length before: %d' % len(records))

    empty_records = []
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
    YelpReviewsPreprocessor.lemmatize_reviews(records)


def plot_roc_curve(y_true, y_predictions):

    false_positive_rate, true_positive_rate, thresholds =\
        roc_curve(y_true, y_predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print('roc auc', roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()

    metrics = {'roc_auc': roc_auc}

    return metrics

# plot_roc_curve(y_test, preds)


def print_confusion_matrix(y_true, y_predictions):

    target_names = ['generic', 'specific']
    print(classification_report(
        y_true, y_predictions, target_names=target_names))

    cm = confusion_matrix(y_true, y_predictions)
    numpy.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)

    # plt.figure()
    # plot_confusion_matrix(cm, title='Unnormalized confusion matrix')
    # plt.show()

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    min_tp_tn = min(cm_normalized[0][0], cm_normalized[1][1])
    print('min of true positives and true negatives: %f' % min_tp_tn)

    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # plt.show()

    accuracy = skmetrics.accuracy_score(y_true, y_predictions)
    precision = skmetrics.precision_score(y_true, y_predictions)
    recall = skmetrics.recall_score(y_true, y_predictions)
    f1 = skmetrics.f1_score(y_true, y_predictions)

    metrics = {
        'dataset': Constants.ITEM_TYPE,
        'max_sentences': Constants.MAX_SENTENCES,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'min_tp_tn': min_tp_tn
    }

    return metrics


def test_classifier(x_matrix, y_vector, sampler_type, my_classifier):
    topic_model_creator.plant_seeds()

    results = {
        'resampler': sampler_type,
        'classifier': type(my_classifier).__name__
    }
    resampled_x, resampled_y = resample(x_matrix, y_vector, sampler_type)
    print('num samples: %d' % len(resampled_y))

    y_predictions, y_true_values = cross_validation_predict(
        my_classifier, resampled_x, resampled_y, 10, sampler_type, 'predict')
    results.update(print_confusion_matrix(y_true_values, y_predictions))

    y_probabilities, y_true_values = cross_validation_predict(
        my_classifier, resampled_x, resampled_y, 10, sampler_type,
        'predict_proba'
    )
    y_probabilities = y_probabilities[:, 1]

    results.update(plot_roc_curve(y_true_values, y_probabilities))

    return results


def cross_validation_predict(
        classifier, x_matrix, y_vector, num_folds, sampler_type, method='predict'):
    cv = StratifiedKFold(y_vector, num_folds)

    all_predictions = []
    all_true_values = []

    for train, test in cv:

        x_train = x_matrix[train]
        y_train = y_vector[train]
        x_test = x_matrix[test]
        y_test = y_vector[test]
        x_train, y_train = resample(x_train, y_train, sampler_type)

        classifier.fit(x_train, y_train)

        func = getattr(classifier, method)
        fold_predictions = func(x_test)
        # fold_predictions = classifier.predict(x_test)
        all_predictions.append(fold_predictions)
        all_true_values.append(y_test)

    predictions = numpy.concatenate(all_predictions)
    true_values = numpy.concatenate(all_true_values)
    return predictions, true_values


def main():
    topic_model_creator.plant_seeds()

    my_resamplers = [
        None,
        'random_over_sampler',
        'smote_regular',
        'smote_bl1',
        'smote_bl2',
        'smote_tomek',
        'smoteenn'
    ]

    my_classifiers = [
        DummyClassifier(strategy='most_frequent', random_state=0),
        DummyClassifier(strategy='stratified', random_state=0),
        DummyClassifier(strategy='uniform', random_state=0),
        DummyClassifier(strategy='constant', random_state=0, constant=True),
        LogisticRegression(C=100),
        SVC(C=1.0, kernel='rbf', probability=True),
        SVC(C=1.0, kernel='linear', probability=True),
        KNeighborsClassifier(n_neighbors=10),
        tree.DecisionTreeClassifier(),
        NuSVC(probability=True),
        RandomForestClassifier(n_estimators=100)
    ]

    max_sentences_list = [None, 1]

    num_cyles = len(my_resamplers) * len(my_classifiers) * len(max_sentences_list)
    index = 1

    results_list = []

    for max_sentences in max_sentences_list:

        Constants.MAX_SENTENCES = max_sentences
        my_records = load_records()
        preprocess_records(my_records)
        x_matrix, y_vector = transform(my_records)

        count_specific_generic(my_records)

        for resampler, classifier in itertools.product(my_resamplers, my_classifiers):

            print('Cycle %d/%d' % (index, num_cyles))

            classification_results =\
                test_classifier(x_matrix, y_vector, resampler, classifier)
            results_list.append(classification_results)
            index += 1

    for results in results_list:
        print(results)

    csv_file = Constants.DATASET_FOLDER + Constants.ITEM_TYPE +\
               '_sentence_classifier_results.csv'
    ETLUtils.save_csv_file(csv_file, results_list, results_list[0].keys())

# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)


