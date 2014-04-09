import nltk
import numpy
import sklearn.cluster as skcluster
import sklearn.metrics as skmetrics
import time
import scipy.cluster as scicluster
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance

__author__ = 'franpena'


class Clusterer:
    def __init__(self):
        pass

    # OK
    @staticmethod
    def k_means_scikit(matrix):
        k_means = skcluster.KMeans(n_clusters=50, init='k-means++',
                                   n_init=1, verbose=1)
        k_means.fit(matrix)

        return k_means.labels_

    # OK
    # With cosine distance the algorithm doesn't converge
    @staticmethod
    def k_means_nltk(matrix):
        clusterer = nltk.KMeansClusterer(50, nltk.euclidean_distance,
                                         avoid_empty_clusters=True,
                                         conv_test=10)
        labels = numpy.array(clusterer.cluster(matrix, True, trace=True))

        return labels

    # OK
    @staticmethod
    def gaac(matrix):
        clusterer = nltk.GAAClusterer()
        labels = numpy.array(clusterer.cluster(matrix, False, trace=True))
        dendrogram = clusterer.dendrogram()
        # dendrogram.show()

        return labels

    # OK
    @staticmethod
    def k_means_scipy(matrix):
        centroids, distortion = scicluster.vq.kmeans(matrix, 50, thresh=0.1)

        print('Centroids:', centroids)
        print('Distortion:', distortion)

    @staticmethod
    def linkage(matrix):
        linkage_matrix = scicluster.hierarchy.linkage(matrix)

        print('Linkage matrix:', linkage_matrix)

        dendrogram = scicluster.hierarchy.dendrogram(linkage_matrix)
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        plt.show()

        leaves = dendrogram['leaves']
        print(leaves)

    @staticmethod
    def mean_shift(matrix):
        mean_shift = skcluster.MeanShift()
        mean_shift.fit(matrix)

        labels = mean_shift.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters:', n_clusters_)

        return labels

    @staticmethod
    def ward(matrix):
        ward = skcluster.Ward(n_clusters=50, compute_full_tree=False)
        ward.fit(matrix)

        return ward.labels_

    @staticmethod
    def dbscan(matrix):
        dbscan = skcluster.DBSCAN(eps=0.3, min_samples=50, metric='euclidean')
        # dbscan = skcluster.DBSCAN(eps=0.3, min_samples=50,
        #                           metric=nltk.cosine_distance)
        dbscan.fit(matrix)

        labels = dbscan.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters:', n_clusters_)

        return labels

    # OK
    @staticmethod
    def evaluate_performance(data, labels, metric='euclidean'):
        score = skmetrics.silhouette_score(data, labels, metric=metric)
        print('Labels:', labels)
        print('Score:', score)

        return score

    @staticmethod
    def cluster_data(matrix, algorithm):

        if algorithm == 'k-means-scikit':
            return Clusterer.k_means_scikit(matrix)
        elif algorithm == 'k-means-nltk':
            return Clusterer.k_means_nltk(matrix)
        # elif algorithm == 'k-means-scipy':
        #     return BusinessClusterer.k_means_scipy(matrix)
        elif algorithm == 'mean-shift':
            return Clusterer.mean_shift(matrix)
        elif algorithm == 'ward':
            return Clusterer.ward(matrix)
        elif algorithm == 'dbscan':
            return Clusterer.dbscan(matrix)

    @staticmethod
    def cluster_and_evaluate_data(matrix, algorithm, metric='euclidean'):
        print('Clustering started using ' + algorithm)
        clustering_start = time.time()
        labels = Clusterer.cluster_data(matrix, algorithm)
        clustering_total = time.time() - clustering_start
        print('Clustering time:', clustering_total)

        evaluation_start = time.time()
        score = Clusterer.evaluate_performance(matrix, labels, metric)
        evaluation_total = time.time() - evaluation_start
        print('Evaluation time:', evaluation_total)
        total = time.time() - clustering_start
        print('Total time:', total)