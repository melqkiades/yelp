import math
import numpy
from scipy import spatial

__author__ = 'fpena'


def chebyshev(vector1, vector2):
    return 1. / (1 + spatial.distance.chebyshev(vector1, vector2))


def cosine(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)


def euclidean(vector1, vector2):
    return 1. / (1 + spatial.distance.euclidean(vector1, vector2))


def manhattan(vector1, vector2):
    return 1. / (1 + spatial.distance.cityblock(vector1, vector2))


def calculate_similarity(vector1, vector2, similarity_metric='euclidean'):

    if similarity_metric == 'chebyshev':
        return chebyshev(vector1, vector2)
    if similarity_metric == 'cosine':
        return cosine(vector1, vector2)
    if similarity_metric == 'euclidean':
        return euclidean(vector1, vector2)
    if similarity_metric == 'manhattan':
        return manhattan(vector1, vector2)
    if similarity_metric == 'pearson':
        similarity_value = numpy.corrcoef(vector1, vector2)[0, 1]
        if similarity_value <= 0 or math.isnan(similarity_value):
            return None
        return similarity_value

    msg = 'Unrecognized similarity metric \'' + similarity_metric + '\''
    raise ValueError(msg)
