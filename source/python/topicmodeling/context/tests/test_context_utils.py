from collections import Counter
import math
import numpy
import numpy.testing as nptest
from topicmodeling.context import context_utils
from topicmodeling.context.review import Review

__author__ = 'fpena'

from unittest import TestCase

empty_paragraph = ""
paragraph1 =\
    "Good morning Dr. Adams. The patient is waiting for you in room number 3."
review_text1 = "We had dinner there last night. The food was delicious. " \
          "Definitely, is the best restaurant in town."
review_text2 = "Small bar, good music, good beer, bad food"


class TestContextUtils(TestCase):

    def test_log_sentences(self):

        expected_value = math.log(2)
        actual_value = context_utils.log_sentences(empty_paragraph)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(3)
        actual_value = context_utils.log_sentences(paragraph1)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(4)
        actual_value = context_utils.log_sentences(review_text1)
        self.assertEqual(actual_value, expected_value)

    def test_log_words(self):

        expected_value = math.log(1)
        actual_value = context_utils.log_words(empty_paragraph)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(15)
        actual_value = context_utils.log_words(paragraph1)
        self.assertEqual(actual_value, expected_value)

        expected_value = math.log(18)
        actual_value = context_utils.log_words(review_text1)
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
        tagged_words = context_utils.tag_words(review_text1)
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
        tagged_words = context_utils.tag_words(review_text1)
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
        tagged_words = context_utils.tag_words(review_text1)
        actual_value = context_utils.get_nouns(tagged_words)
        expected_value = ['dinner', 'night', 'food', 'restaurant', 'town']
        self.assertItemsEqual(actual_value, expected_value)

    def test_get_all_nouns(self):
        reviews = [empty_paragraph, paragraph1, review_text1, review_text2]
        actual_value = context_utils.get_all_nouns(reviews)
        expected_value = set([
            'morning', 'dr', 'adams', 'patient', 'room', 'number', 'dinner',
            'night', 'food', 'restaurant', 'town', 'bar', 'music', 'beer'
        ])
        self.assertEqual(actual_value, expected_value)

    def test_remove_nouns_from_reviews(self):

        nouns = ['bar', 'night', 'food', 'wine']
        actual_review1 = Review(review_text1)
        actual_review2 = Review(review_text2)
        actual_reviews = [actual_review1, actual_review2]

        expected_review1 = Review(review_text1)
        expected_review2 = Review(review_text2)
        expected_review1.nouns.remove('night')
        expected_review1.nouns.remove('food')
        expected_review2.nouns.remove('bar')
        expected_review2.nouns.remove('food')

        context_utils.remove_nouns_from_reviews(actual_reviews, nouns)
        print(actual_reviews[0].nouns)
        print(actual_reviews[1].nouns)
        self.assertItemsEqual(actual_review1.nouns, expected_review1.nouns)
        self.assertItemsEqual(actual_review2.nouns, expected_review2.nouns)

    def test_split_list_by_labels(self):

        lst = ['a', 'b', 'c', 'd', 'e', 'f']
        labels = [2, 0, 0, 1, 1, 0]
        expected_matrix = [
            ['b', 'c', 'f'],
            ['d', 'e'],
            ['a']
        ]
        actual_matrix = context_utils.split_list_by_labels(lst, labels)

        self.assertItemsEqual(actual_matrix, expected_matrix)


        mm = [[1, 2, 3], [4, 5]]
        print(reduce(lambda x, y: x + sum(y), mm, 0))

