from scipy import spatial
from etl import similarity

__author__ = 'fpena'

from unittest import TestCase
import numpy as np


class TestSimilarity(TestCase):

    def test_cosine(self):

        v1 = [4.5, 3]
        v2 = [4.0, 2]
        expected_value = (4.5 * 4 + 3 * 2) / ((4.5**2 + 3**2)**0.5 * (4**2 + 2**2)**0.5)
        self.assertEqual(expected_value, similarity.cosine(v1, v2))

        v1 = [5, 1, 2]
        v2 = [5, 1, 2]
        self.assertEqual(1, similarity.cosine(v1, v2))

        v1 = [1, 2, 3, 4, 5]
        v2 = [1, 22, 3000, 4, 5000000000]
        print(similarity.cosine(v1, v2))
        print(1 / (1 + spatial.distance.cosine(v1, v2)))
        print(spatial.distance.cosine(v1, v2))
        print(1 - (np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))))
        # (4.5 * 4 + 3 * 2) / (4.5**2 + 3**2)**0.5 * (4**2 + 2**2)**0.5

    def test_euclidean(self):

        v1 = [4.5, 3]
        v2 = [4.0, 2]
        expected_value = 0.47213595499957939
        self.assertEqual(expected_value, similarity.euclidean(v1, v2))

        v1 = [5, 1, 2]
        v2 = [5, 1, 2]
        self.assertEqual(1, similarity.euclidean(v1, v2))