import random

import numpy

from utils.constants import Constants


def plant_seeds():

    if Constants.RANDOM_SEED is not None:
        print('random seed: %d' % Constants.RANDOM_SEED)
        random.seed(Constants.RANDOM_SEED)
    if Constants.NUMPY_RANDOM_SEED is not None:
        print('numpy random seed: %d' % Constants.NUMPY_RANDOM_SEED)
        numpy.random.seed(Constants.NUMPY_RANDOM_SEED)
