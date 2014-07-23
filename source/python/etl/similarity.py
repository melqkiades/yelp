from scipy import spatial

__author__ = 'fpena'


def cosine(vector1, vector2):

    numerator = 0.
    vector1_denominator = 0.
    vector2_denominator = 0.

    for value1, value2 in zip(vector1, vector2):
        numerator += value1 * value2
        vector1_denominator += value1 ** 2
        vector2_denominator += value2 ** 2

    denominator = (vector1_denominator ** 0.5) * (vector2_denominator ** 0.5)

    if denominator == 0:
        return None

    similarity = numerator / denominator

    return similarity


def euclidean(vector1, vector2):
    return 1. / (1 + spatial.distance.euclidean(vector1, vector2))
