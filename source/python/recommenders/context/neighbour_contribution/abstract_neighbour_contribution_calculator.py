from abc import ABCMeta

__author__ = 'fpena'


class AbstractNeighbourContributionCalculator:

    __metaclass__ = ABCMeta

    def __init__(self):
        self.user_baseline_calculator = None

    def clear(self):
        self.user_baseline_calculator = None
