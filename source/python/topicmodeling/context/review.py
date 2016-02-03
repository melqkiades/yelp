from topicmodeling.context import review_utils

__author__ = 'fpena'


class Review(object):

    def __init__(self, text=None):
        self.id = None
        self.text = text
        self.tagged_words = None
        self.nouns = None
        if text is not None:
            self.tagged_words = review_utils.tag_words(self.text)
            self.nouns = review_utils.get_nouns(self.tagged_words)
        self.senses = None
        self.topics = None
        self.user_id = None
        self.item_id = None
        self.rating = None
        self.type = None  # Either specific or generic
        self.date = None
