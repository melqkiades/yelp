from topicmodeling.context import context_utils

__author__ = 'fpena'


class Review:

    def __init__(self, text):
        # self.id = id
        self.text = text
        self.tagged_words = context_utils.tag_words(self.text)
        self.nouns = context_utils.get_nouns(self.tagged_words)
        self.senses = None

