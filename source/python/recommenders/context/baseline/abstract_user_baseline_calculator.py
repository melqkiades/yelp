from abc import ABCMeta, abstractmethod

__author__ = 'fpena'


class AbstractUserBaselineCalculator:

    __metaclass__ = ABCMeta

    def __init__(self):
        self.user_dictionary = None
        self.topic_indices = None

    def load(self, user_dictionary, topic_indices):
        self.user_dictionary = user_dictionary
        self.topic_indices = topic_indices

    @abstractmethod
    def calculate_user_baseline(self, user_id, context, threshold):
        """

        """

    @abstractmethod
    def get_rating_on_context(self, user_id, item_id, context, threshold):
        """

        """

    def clear(self):
        self.user_dictionary = None
        self.topic_indices = None
