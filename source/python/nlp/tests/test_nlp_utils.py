from collections import Counter

from nlp import nlp_utils
from unittest import TestCase


__author__ = 'fpena'


empty_paragraph = ""
paragraph1 =\
    "Good morning Dr. Adams. The patient is waiting for you in room number 3."
paragraph2 = "I'm going to my parent's place over the summer! I can't wait " \
             "to get there"
review_text1 = "We had dinner there last night. The food was delicious. " \
          "Definitely, is the best restaurant in town."
review_text2 = "Small bar, good music, good beer, bad food"
review_text6 = "My first trip to Phoenix couldn't have been better. I went " \
               "to JW for 3 nights and loved every minute of it. It's " \
               "beautiful, the pools (about 6 of them) were very nice, and " \
               "the rooms were very spacious.\nThe ONLY thing I didn't like " \
               "was the parking. It's really far away from the lobby/check " \
               "in area. So if you have a lot of stuff to bring in, try to " \
               "up front. \nOther than that, I experienced nothing even " \
               "slightly bad at JW.\n\nWould definitely go back again!"
review_text9 = "Beef gyros are always good here."


class TestNlpUtils(TestCase):

    def test_get_sentences(self):

        actual_value = nlp_utils.get_sentences(empty_paragraph)
        expected_value = []
        self.assertEqual(actual_value, expected_value)
        actual_value = nlp_utils.get_sentences(paragraph1)
        expected_value = [
            'Good morning Dr. Adams.',
            'The patient is waiting for you in room number 3.'
        ]
        self.assertEqual(actual_value, expected_value)
        actual_value = len(nlp_utils.get_sentences(review_text6))
        expected_value = 8
        self.assertEqual(actual_value, expected_value)

    def test_get_words(self):
        actual_value = nlp_utils.get_words(empty_paragraph)
        expected_value = []
        self.assertEqual(actual_value, expected_value)
        actual_value = nlp_utils.get_words(paragraph1)
        expected_value = [
            'Good', 'morning', 'Dr.', 'Adams', '.',
            'The', 'patient', 'is', 'waiting', 'for', 'you', 'in', 'room',
            'number', '3', '.'
        ]
        self.assertEqual(actual_value, expected_value)
        actual_value = len(nlp_utils.get_words(paragraph2))
        expected_value = 19
        self.assertEqual(actual_value, expected_value)
        self.assertEqual(actual_value, expected_value)
        actual_value = len(nlp_utils.get_words(review_text6))
        expected_value = 106
        self.assertEqual(actual_value, expected_value)

    def test_tag_words(self):
        actual_value = nlp_utils.tag_words(empty_paragraph)
        expected_value = []
        self.assertEqual(actual_value, expected_value)
        actual_value = nlp_utils.tag_words(paragraph1)
        expected_value = [
            ('good', 'JJ'), ('morning', 'NN'), ('dr.', 'NN'), ('adams', 'NN'),
            ('.', '.'), ('the', 'DT'), ('patient', 'NN'), ('is', 'VBZ'),
            ('waiting', 'VBG'), ('for', 'IN'), ('you', 'PRP'), ('in', 'IN'),
            ('room', 'NN'), ('number', 'NN'), ('3', 'CD'), ('.', '.')
        ]
        self.assertEqual(actual_value, expected_value)
        actual_value = nlp_utils.tag_words(paragraph2)
        expected_value = [
            ('i', 'NN'), ("'m", 'VBP'), ('going', 'VBG'), ('to', 'TO'),
            ('my', 'PRP$'), ('parent', 'NN'), ("'s", 'POS'),
            ('place', 'NN'), ('over', 'IN'), ('the', 'DT'),
            ('summer', 'NN'), ('!', '.'),
            ('i', 'NN'), ('ca', 'MD'), ("n't", 'RB'), ('wait', 'VB'),
            ('to', 'TO'), ('get', 'VB'), ('there', 'RB')
        ]
        self.assertEqual(actual_value, expected_value)

    def test_count_verbs(self):
        tagged_words = nlp_utils.tag_words(empty_paragraph)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = nlp_utils.count_verbs(counts)
        expected_value = 0
        self.assertEqual(actual_value, expected_value)
        tagged_words = nlp_utils.tag_words(paragraph1)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = nlp_utils.count_verbs(counts)
        expected_value = 2
        self.assertEqual(actual_value, expected_value)
        tagged_words = nlp_utils.tag_words(paragraph2)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = nlp_utils.count_verbs(counts)
        expected_value = 4
        self.assertEqual(actual_value, expected_value)
