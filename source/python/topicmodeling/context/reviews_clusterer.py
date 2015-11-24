from collections import Counter
from pandas import DataFrame, pandas
from sklearn import neighbors
import string
import math
from nltk import tokenize
import nltk
import numpy as np
import cPickle as pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score, \
    adjusted_mutual_info_score, homogeneity_score
from etl import ETLUtils
from topicmodeling.context import review_metrics_extractor
from topicmodeling.context import context_utils
from topicmodeling.context.review import Review
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


NUM_FEATURES = 2


def cluster_reviews(reviews):
    """
    Classifies a list of reviews into specific and generic. Returns a list of
    integer of the same size as the list of reviews, in which each position of
    the list contains a 0 if that review is specific or a 1 if that review is
    generic.

    :param reviews: a list of reviews. Each review must contain the text of the
    review and the part-of-speech tags for every word
    :type reviews: list[Review]
    :return a list of integer of the same size as the list of reviews, in which
    each position of the list contains a 0 if that review is specific or a 1 if
    that review is generic
    """

    records = np.zeros((len(reviews), NUM_FEATURES))

    for index in range(len(reviews)):
        records[index] =\
            review_metrics_extractor.get_review_metrics(reviews[index])
    review_metrics_extractor.normalize_matrix_by_columns(records)

    k_means = KMeans(n_clusters=2)
    k_means.fit(records)
    labels = k_means.labels_

    record_clusters = split_list_by_labels(records, labels)
    cluster0_sum = reduce(lambda x, y: x + sum(y), record_clusters[0], 0)
    cluster1_sum = reduce(lambda x, y: x + sum(y), record_clusters[1], 0)

    if cluster0_sum < cluster1_sum:
        # If the cluster 0 contains the generic review we invert the tags
        labels = [1 if element == 0 else 0 for element in labels]

    return labels


def split_list_by_labels(lst, labels):
    """
    Receives a list of objects and a list of labels (each label is an integer
    number) and returns a matrix in which all the elements have been
    grouped by label. For instance if we have lst = ['a', 'b', 'c', 'd', 'e']
    and labels = [0, 0, 1, 1, 0] then the result of calling this function would
    be matrix = [['a', 'b', 'e'], ['c', 'd']]. This is particularly useful when
    we call matrix[0] = ['a', 'b', 'e'] or matrix[1] = ['c', 'd']

    :type lst: numpy.array
    :param lst: a list of objects
    :type labels: list[int]
    :param labels: a list of integer with the label for each element of lst
    :rtype: list[list[]]
    :return:
    """
    matrix = []

    for index in range(max(labels) + 1):
        matrix.append([])

    for index in range(len(labels)):
        element = lst[index]
        matrix[labels[index]].append(element)

    return matrix


def get_stats_from_reviews(reviews):

    records = np.zeros((len(reviews), 5))

    for index in range(len(reviews)):
        records[index] = count_review_info(reviews[index])

    max_values = records.max(axis=0)
    min_values = records.min(axis=0)
    mean_values = records.mean(axis=0)

    stats = {
        'total_reviews': len(reviews),
        'sentences': {'max': max_values[0], 'min': min_values[0], 'mean': mean_values[0]},
        'words': {'max': max_values[1], 'min': min_values[1], 'mean': mean_values[1]},
        'past_verbs': {'max': max_values[2], 'min': min_values[2], 'mean': mean_values[2]},
        'verbs': {'max': max_values[3], 'min': min_values[3], 'mean': mean_values[3]},
        'ratio': {'max': max_values[4], 'min': min_values[4], 'mean': mean_values[4]},
    }

    return stats


def count_review_info(review):
    num_sentences = len(review_metrics_extractor.get_sentences(review.text))
    num_words = len(review_metrics_extractor.get_words(review.text))
    tagged_words = review.tagged_words
    counts = Counter(tag for word, tag in tagged_words)
    num_past_verbs = float(counts['VBD'])
    num_verbs = review_metrics_extractor.count_verbs(counts)

    # This ensures that when log_verbs = 0 the program won't crash
    if num_verbs == 0:
        verbs_ratio = 0
    else:
        verbs_ratio = num_past_verbs / num_verbs

    result = [num_sentences, num_words, num_past_verbs, num_verbs, verbs_ratio]

    # print('ratio', verbs_ratio, '\tpast verbs', num_past_verbs, 'verbs', num_verbs)

    return np.array(result)


def compare_clusterers():

    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    # my_file = my_folder + 'classified_hotel_reviews.json'
    # binary_reviews_file = my_folder + 'classified_hotel_reviews.pkl'
    my_file = my_folder + 'classified_restaurant_reviews.json'
    binary_reviews_file = my_folder + 'classified_restaurant_reviews.pkl'
    my_records = ETLUtils.load_json_file(my_file)
    my_labels = np.array([record['specific'] == 'yes' for record in my_records])

    # my_reviews = build_reviews(my_records)
    # with open(binary_reviews_file, 'wb') as write_file:
    #     pickle.dump(my_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    with open(binary_reviews_file, 'rb') as read_file:
        my_reviews = pickle.load(read_file)

    my_metrics = np.zeros((len(my_reviews), NUM_FEATURES))
    for index in range(len(my_reviews)):
        my_metrics[index] =\
            review_metrics_extractor.get_review_metrics(my_reviews[index])
        # print(my_metrics[index])

    review_metrics_extractor.normalize_matrix_by_columns(my_metrics)

    clusterers = [
        KMeans(n_clusters=2),
        MiniBatchKMeans(n_clusters=4),
        # MeanShift(),
        # AgglomerativeClustering(n_clusters=2)
        # DummyClassifier(strategy='most_frequent', random_state=0),
        # DummyClassifier(strategy='stratified', random_state=0),
        # DummyClassifier(strategy='uniform', random_state=0),
        # DummyClassifier(strategy='constant', random_state=0, constant=True),
        # LogisticRegression(C=100),
        # SVC(C=1.0, kernel='rbf'),
        # SVC(C=1.0, kernel='linear'),
        # KNeighborsClassifier(n_neighbors=10),
        # tree.DecisionTreeClassifier(),
        # NuSVC(),
        # LinearSVC()
    ]
    scores = [[] for i in range(len(clusterers))]

    Xtrans = my_metrics
    # pca = decomposition.PCA(n_components=2)
    # Xtrans = pca.fit_transform(my_metrics)
    # print(pca.explained_variance_ratio_)
    # lda_inst = lda.LDA()
    # Xtrans = lda_inst.fit_transform(my_metrics, my_labels)
    # print(lda_inst.get_params())
    # mds = manifold.MDS(n_components=2)
    # Xtrans = mds.fit_transform(my_metrics)

    cv = KFold(n=len(my_metrics), n_folds=5)

    for i in range(len(clusterers)):
        for train, test in cv:
            x_train, y_train = Xtrans[train], my_labels[train]
            x_test, y_test = Xtrans[test], my_labels[test]

            clusterer = clusterers[i]
            clusterer.fit(x_train)
            cluster_labels = clusterer.predict(x_test)

            scores_map = {}
            scores_map['silhouette'] = silhouette_score(x_test, cluster_labels)
            scores_map['adjusted_rand'] = adjusted_rand_score(y_test, cluster_labels)
            scores_map['adjusted_mutual'] = adjusted_mutual_info_score(y_test, cluster_labels)
            scores_map['homogeneity'] = homogeneity_score(y_test, cluster_labels)
            scores[i].append(scores_map)
            # print('silhouette score', silhouette_avg)

    for score in scores:
        silhouette = [d['silhouette'] for d in score]
        adjusted_rand = [d['adjusted_rand'] for d in score]
        adjusted_mutual = [d['adjusted_mutual'] for d in score]
        homogeneity = [d['homogeneity'] for d in score]
        print("Silhouette:\t\t\tMean=%.5f\tStddev=%.5f" %
              (np.mean(silhouette), np.std(silhouette)))
        print("Adjusted Rand:\t\tMean=%.5f\tStddev=%.5f" %
              (np.mean(adjusted_rand), np.std(adjusted_rand)))
        print("Adjusted Mutual:\tMean=%.5f\tStddev=%.5f" %
              (np.mean(adjusted_mutual), np.std(adjusted_mutual)))
        print("Homogeneity:\t\tMean=%.5f\tStddev=%.5f" %
              (np.mean(homogeneity), np.std(homogeneity)))


# my_file = '/Users/fpena/tmp/reviews_restaurant_shuffled.pkl'
# # my_file = '/Users/fpena/tmp/reviews_hotel_shuffled.pkl'
# # my_file = '/Users/fpena/tmp/reviews_spa.pkl'
# with open(my_file, 'rb') as read_file:
#     my_reviews = pickle.load(read_file)
#
# print(get_stats_from_reviews(my_reviews))


def main():
    # my_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_hotel_reviews.json'
    my_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_restaurant_reviews.json'
    my_records = ETLUtils.load_json_file(my_file)
    # my_reviews = []
    # my_index = 0
    #
    # print("records:", len(my_records))
    #
    # for record in my_records:
    #     my_index += 1
    #     my_reviews.append(Review(record['text']))
    #     print('index', my_index)

    # binary_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_hotel_reviews.pkl'
    binary_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_restaurant_reviews.pkl'
    # with open(binary_reviews_file, 'wb') as write_file:
    #     pickle.dump(my_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    with open(binary_reviews_file, 'rb') as read_file:
        my_reviews = pickle.load(read_file)

    # knn = cluster_reviews2(my_reviews, my_records)



    cluster_labels = cluster_reviews(my_reviews)
    specific_records = split_list_by_labels(my_records, cluster_labels)[0]
    generic_records = split_list_by_labels(my_records, cluster_labels)[1]


    specific_precision = 0.0
    for record in specific_records:
        if record['context'] == 'yes':
            specific_precision += 1

    specific_precision /= len(specific_records)
    print('context precision', specific_precision)

    generic_precision = 0.0
    for record in generic_records:
        if record['context'] == 'no':
            generic_precision += 1

    generic_precision /= len(generic_records)
    print('context precision', generic_precision)

    compare_clusterers()

# start = time.tif seconds" % total_time)

# folder = '/Users/fp\-oupby(['user_id']).size().order(ascending=False))
# data_frame.sort('user_id', ascending=False, inplace=True )
# print(data_frame.groupby('user_id', sort=False).sum())


# my_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_hotel_reviews.json'
# my_records = ETLUtils.load_json_file(my_file)
# binary_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_hotel_reviews.pkl'
# with open(binary_reviews_file, 'rb') as read_file:
#     my_reviews = pickle.load(read_file)
# my_metrics = np.zeros((len(my_reviews), 5))
# for index in range(len(my_reviews)):
#     my_metrics[index] = get_review_metrics(my_reviews[index])
# data_frame = DataFrame(my_records, columns=['context'])
# pandas.set_option('display.max_rows', len(data_frame))
# print(data_frame)
#
# # for row, metrics in zip(data_frame.iterrows(), my_metrics):
# #     print(row, metrics)
# #
# print(data_frame.groupby(['context']).size())
#
#
#
# cluster_labels = cluster_reviews(my_reviews)
# specific_records = split_list_by_labels(my_records, cluster_labels)[0]
# generic_records = split_list_by_labels(my_records, cluster_labels)[1]
#
# num_context_specific = 0.0
# num_no_context_specific = 0.0
# for record in specific_records:
#     if record['context'] == 'yes':
#         num_context_specific += 1
#     else:
#         num_no_context_specific += 1
#
# num_context_specific_pctg = num_context_specific / len(specific_records)
# num_no_context_specific_pctg = num_no_context_specific/len(specific_records)
# print('specific total', len(specific_records))
# print('specific context', num_context_specific, num_context_specific_pctg)
# print('specific no context', num_no_context_specific, num_no_context_specific_pctg)
#
#
# num_context_generic = 0.0
# num_no_context_generic = 0.0
# for record in generic_records:
#     if record['context'] == 'yes':
#         num_context_generic += 1
#     else:
#         num_no_context_generic += 1
#
# num_context_generic_pctg = num_context_generic / len(generic_records)
# num_no_context_generic_pctg = num_no_context_generic/len(generic_records)
# print('generic total', len(generic_records))
# print('generic context', num_context_generic, num_context_generic_pctg)
# print('generic no context', num_no_context_generic, num_no_context_generic_pctg)