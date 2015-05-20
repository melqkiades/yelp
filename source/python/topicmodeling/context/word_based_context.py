from nltk.corpus import wordnet
from topicmodeling.context import context_utils
from topicmodeling.context.review import Review

__author__ = 'fpena'


class WordBasedContext:

    def __init__(self, text_reviews):
        self.text_reviews = text_reviews
        self.alpha = 0.005
        self.beta = 1.0
        self.reviews = None
        self.all_nouns = None

    def init_reviews(self):
        self.reviews = []

        for text_review in self.text_reviews:
            self.reviews.append(Review(text_review))

        self.all_nouns = context_utils.get_all_nouns(self.reviews)

    def filter_nouns(self):
        specific_reviews = []
        generic_reviews = []

        for noun in self.all_nouns:
            specific_weighted_frq =\
                context_utils.calculate_weighted_frequency(noun, specific_reviews)
            generic_weighted_frq =\
                context_utils.calculate_weighted_frequency(noun, generic_reviews)

            if generic_weighted_frq < self.alpha or specific_weighted_frq < self.alpha:
                self.all_nouns.remove(noun)
                continue

            ratio = specific_weighted_frq / generic_weighted_frq

            if ratio < self.beta:
                self.all_nouns.remove(noun)
                continue

    def build_groups(self, noun):
        synsets = wordnet.synsets(noun, pos='n')

        for synset in synsets:

            # Compare every element with the rest of the list, and then if
            # the nouns are similar, create a group
            pass








