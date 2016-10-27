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


def generate_file_name(
        name, extension, folder, cycle_index, fold_index, uses_context):

    prefix = Constants.ITEM_TYPE + '_' + name + '_' + \
             Constants.TOPIC_MODEL_TYPE
    context_suffix = ''
    if uses_context:
        context_suffix = \
            '_numtopics:' + str(Constants.TOPIC_MODEL_NUM_TOPICS) + \
            '_iterations:' + str(Constants.TOPIC_MODEL_ITERATIONS) + \
            '_passes:' + str(Constants.TOPIC_MODEL_PASSES) + \
            '_bow:' + str(Constants.BOW_TYPE)
    suffix = context_suffix + \
        '_reviewtype:' + str(Constants.TOPIC_MODEL_REVIEW_TYPE) + \
        '_document_level:' + str(Constants.DOCUMENT_LEVEL) + \
        '.' + extension

    if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
        topic_model_file = prefix + '_separated' + suffix
    elif cycle_index is None and fold_index is None:
        topic_model_file = prefix + '_full' + suffix
    else:
        strategy = Constants.CROSS_VALIDATION_STRATEGY
        cross_validation_info = '_' + strategy
        if strategy == 'nested_validate':
            cross_validation_info +=\
                ':' + str(Constants.NESTED_CROSS_VALIDATION_CYCLE)
        topic_model_file = prefix + \
            cross_validation_info + \
            '_cycle:' + str(cycle_index+1) + '|' + str(Constants.NUM_CYCLES) + \
            '_fold:' + str(fold_index+1) + '|' + \
            str(Constants.CROSS_VALIDATION_NUM_FOLDS) + \
            suffix
    return folder + topic_model_file
