from topicmodeling.context import review_utils

__author__ = 'fpena'


class Review:

    def __init__(self, text):
        # self.id = id
        self.text = text
        self.tagged_words = review_utils.tag_words(self.text)
        self.nouns = review_utils.get_nouns(self.tagged_words)
        self.senses = None
        self.topics = None

