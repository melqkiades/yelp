from abc import ABCMeta, abstractmethod

__author__ = 'fpena'


class AbstractNeighbourhoodCalculator:

    __metaclass__ = ABCMeta

    def __init__(self):
        self.user_ids = None
        self.user_dictionary = None
        self.topic_indices = None
        self.num_neighbours = None
        self.similarity_matrix = None

    @abstractmethod
    def load(self, user_ids, user_dictionary, topic_indices, num_neighbours):
        """
        Loads the values into the class
        """

    @abstractmethod
    def get_neighbourhood(self, user, item, context, threshold):
        """
        Calculates the neighbourhood of the given user
        """

    @abstractmethod
    def clear(self):
        """
        Clears the variables of this object to free memory
        """
