from collections import Counter
import math
import numpy
import numpy.testing as nptest
from topicmodeling.context import context_utils

__author__ = 'fpena'

from unittest import TestCase

empty_paragraph = ""
paragraph1 =\
    "Good morning Dr. Adams. The patient is waiting for you in room number 3."
review1 = "We had dinner there last night. The food was delicious. " \
          "Definitely, is the best restaurant in town."
review2 = "Small bar, good music, good beer, bad food"


class TestContextUtils(TestCase):

    def test_log_sentences(self):

        expected_value = math.log(2)
        actual_value = context_utils.log_sentences(empty_paragraph)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(3)
        actual_value = context_utils.log_sentences(paragraph1)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(4)
        actual_value = context_utils.log_sentences(review1)
        self.assertEqual(actual_value, expected_value)

    def test_log_words(self):

        expected_value = math.log(1)
        actual_value = context_utils.log_words(empty_paragraph)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(15)
        actual_value = context_utils.log_words(paragraph1)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(18)
        actual_value = context_utils.log_words(review1)
        self.assertEqual(actual_value, expected_value)

    def test_vbd_sum(self):

        expected_value = math.log(1)
        tagged_words = context_utils.tag_words(empty_paragraph)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = context_utils.vbd_sum(counts)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(1)
        tagged_words = context_utils.tag_words(paragraph1)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = context_utils.vbd_sum(counts)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(3)
        tagged_words = context_utils.tag_words(review1)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = context_utils.vbd_sum(counts)
        self.assertEqual(actual_value, expected_value)

    def test_verb_sum(self):

        expected_value = math.log(1)
        tagged_words = context_utils.tag_words(empty_paragraph)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = context_utils.verb_sum(counts)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(3)
        tagged_words = context_utils.tag_words(paragraph1)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = context_utils.verb_sum(counts)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(4)
        tagged_words = context_utils.tag_words(review1)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = context_utils.verb_sum(counts)
        self.assertEqual(actual_value, expected_value)

    def test_process_review(self):

        expected_value = numpy.array([math.log(2), 0, 0, 0, 1])
        actual_value = context_utils.process_review(empty_paragraph)
        nptest.assert_allclose(actual_value, expected_value)

    def test_get_nouns(self):

        tagged_words = context_utils.tag_words(empty_paragraph)
        actual_value = context_utils.get_nouns(tagged_words)
        expected_value = []
        self.assertItemsEqual(actual_value, expected_value)
        tagged_words = context_utils.tag_words(paragraph1)
        actual_value = context_utils.get_nouns(tagged_words)
        expected_value = ['morning', 'dr', 'adams', 'patient', 'room', 'number']
        self.assertItemsEqual(actual_value, expected_value)
        tagged_words = context_utils.tag_words(review1)
        actual_value = context_utils.get_nouns(tagged_words)
        expected_value = ['dinner', 'night', 'food', 'restaurant', 'town']
        self.assertItemsEqual(actual_value, expected_value)

    def test_get_all_nouns(self):
        reviews = [empty_paragraph, paragraph1, review1, review2]
        actual_value = context_utils.get_all_nouns(reviews)
        expected_value = set([
            'morning', 'dr', 'adams', 'patient', 'room', 'number', 'dinner',
            'night', 'food', 'restaurant', 'town', 'bar', 'music', 'beer'
        ])
        self.assertEqual(actual_value, expected_value)
