# import sys
# sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')

import itertools

import numpy
import time
from sklearn.cluster import AffinityPropagation

from etl import ETLUtils


class BaumanSensesGrouper(object):

    def __init__(self, similarity_matrix, threshold):
        self.similarity_matrix = similarity_matrix
        self.threshold = threshold
        self.counter = 0

    def group_senses(self):

        senses = range(len(self.similarity_matrix))
        group_list = []

        for sense in senses:
            candidate_groups = []
            new_group = {sense}
            # group_list.append()

            for old_group in group_list:
                if self.can_combine_groups(old_group, new_group):
                    candidate_groups.append(old_group)
            candidate_groups.append(new_group)
            group_list.append(new_group)

            for candidate_group1, candidate_group2 in itertools.combinations(
                    candidate_groups, 2):
                if self.can_combine_groups(
                        candidate_group1, candidate_group2):
                    self.combine_groups(
                        candidate_group1, candidate_group2)
                    if candidate_group2 in group_list:
                        group_list.remove(candidate_group2)

        print('Total iterations for grouping senses: %d' % self.counter)

        return group_list

    def calculate_similarity(self, element1, element2):
        if self.similarity_matrix[element1][element2] >= self.threshold:
            return True
        return False

    def can_combine_groups(self, group1, group2):

        for element1 in group1:
            for element2 in group2:
                self.counter += 1
                if not self.calculate_similarity(element1, element2):
                    # print('false', element1, element2)
                    # print('similarity', self.calculate_similarity(element1, element2))
                    # print(self.similarity_matrix[element1, element2])
                    # print('threshold: %f', self.threshold)
                    return False
        return True

    @staticmethod
    def combine_groups(group1, group2):
        group1 |= group2
        return group1


def cluster_affinity_propagation(similarity_matrix, desired_keys=None):

    numpy_matrix = similarity_matrix_to_numpy(similarity_matrix, desired_keys)

    clusterer = AffinityPropagation()
    return clusterer.fit_predict(numpy_matrix)


def similarity_matrix_to_numpy(similarity_matrix, desired_keys=None):
    size = len(desired_keys)
    numpy_matrix = numpy.ndarray((size, size))
    keys = sorted(similarity_matrix)
    keys = [key for key in keys if key in desired_keys]

    for index1 in range(len(keys)):
        key1 = keys[index1]
        for index2 in range(len(keys)):
            key2 = keys[index2]
            numpy_matrix[index1][index2] = similarity_matrix[key1][key2]

    return numpy_matrix


# def similarity_matrix_to_numpy(similarity_matrix, desired_keys=None):
#     # size = len(similarity_matrix)
#     size = len(desired_keys)
#     numpy_matrix = numpy.ndarray((size, size))
#     # keys = similarity_matrix.keys()
#     keys = sorted(similarity_matrix)
#     # keys = [key for key in keys if key in desired_keys]
#     # index = 0
#     index1 = 0
#
#     for key1 in keys:
#         if key1 not in desired_keys:
#             continue
#         index2 = 0
#         for key2 in keys:
#             if key2 not in desired_keys:
#                 continue
#             numpy_matrix[index1][index2] = similarity_matrix[key1][key2]
#             index2 += 1
#         index1 += 1
#
#     return numpy_matrix




# my_similarity_matrix = {
#     'a': {'a': 1.0, 'b': 0.8, 'c': 0.5, 'd': 0.3, 'e': 0.9},
#     'b': {'a': 0.8, 'b': 1.0, 'c': 0.8, 'd': 0.6, 'e': 1.0},
#     'c': {'a': 0.5, 'b': 0.8, 'c': 1.0, 'd': 0.9, 'e': 0.2},
#     'd': {'a': 0.3, 'b': 0.6, 'c': 0.9, 'd': 1.0, 'e': 0.7},
#     'e': {'a': 0.9, 'b': 1.0, 'c': 0.2, 'd': 0.7, 'e': 1.0}
# }
#
# bauman_grouper = BaumanSensesGrouper(my_similarity_matrix, 0.8)
# my_senses = ['a', 'b', 'c', 'd', 'e']
# my_groups = bauman_grouper.group_senses(my_senses)
#
# # ETLUtils.add_transpose_list_column()
#
# print(my_groups)
#
# my_desired_keys = desired_keys=['e', 'd', 'c', 'a', 'b']
#
# my_numpy_matrix = similarity_matrix_to_numpy(my_similarity_matrix, my_desired_keys)
# print(my_numpy_matrix)
# print(cluster_affinity_propagation(my_similarity_matrix, my_desired_keys))

# new_group = {4}
# print(new_group)
