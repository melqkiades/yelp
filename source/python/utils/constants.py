import platform
from string import strip

import yaml
import subprocess

__author__ = 'fpena'


CODE_FOLDER = '/home/fpena/yelp/source/python/'
# CODE_FOLDER = '/Users/fpena/UCC/Thesis/projects/yelp/source/python/'
PROPERTIES_FILE = CODE_FOLDER + 'properties.yaml'


def load_properties():
    with open(PROPERTIES_FILE, 'r') as f:
        return yaml.load(f)


class Constants(object):

    # Please keep the constants' names in alphabetical order to avoid problems
    # with the version control system (merging)

    BOW_FIELD = 'bow'
    CONTEXT_TOPICS_FIELD = 'context_topics'
    CONTEXT_WORDS_FIELD = 'context_words'
    CORPUS_FIELD = 'corpus'
    ITEM_ID_FIELD = 'business_id'
    POS_TAGS_FIELD = 'pos_tags'
    PREDICTED_CLASS_FIELD = 'predicted_class'
    RATING_FIELD = 'stars'
    REVIEW_ID_FIELD = 'review_id'
    TEXT_FIELD = 'text'
    TOPICS_FIELD = 'topics'
    USER_ID_FIELD = 'user_id'
    VOTES_FIELD = 'votes'

    SPECIFIC = 'specific'
    GENERIC = 'generic'
    ALL_TOPICS = 'all_topics'
    LIBFM = 'libfm'
    FASTFM = 'fastfm'

    # Folders
    DATASET_FOLDER = '/home/fpena/data/'
    LIBFM_FOLDER = '/home/fpena/libfm-master/bin/'
    # DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/stuff/'
    # LIBFM_FOLDER = '/Users/fpena/tmp/libfm-master/bin/'
    GENERATED_FOLDER = DATASET_FOLDER + 'generated_context/'

    _properties = load_properties()
    ITEM_TYPE = _properties['business_type']
    REVIEW_TYPE = _properties['review_type']
    TOPN_N = _properties['topn_n']
    TOPN_NUM_ITEMS = _properties['topn_num_items']
    RANDOM_SEED = _properties['random_seed']
    NUMPY_RANDOM_SEED = _properties['numpy_random_seed']
    NUM_CYCLES = _properties['num_cycles']
    LDA_ALPHA = _properties['lda_alpha']
    LDA_BETA = _properties['lda_beta']
    LDA_EPSILON = _properties['lda_epsilon']
    LDA_NUM_TOPICS = _properties['lda_num_topics']
    LDA_MODEL_PASSES = _properties['lda_model_passes']
    LDA_MODEL_ITERATIONS = _properties['lda_model_iterations']
    LDA_MULTICORE = _properties['lda_multicore']
    LIBFM_SEED = _properties['libfm_seed']
    FM_NUM_FACTORS = _properties['fm_num_factors']
    CROSS_VALIDATION_NUM_FOLDS =\
        _properties['cross_validation_num_folds']
    SHUFFLE_DATA = _properties['shuffle_data']
    USE_CONTEXT = _properties['use_context']
    NUM_CORES = _properties['num_cores']
    CACHE_TOPIC_MODEL = _properties['cache_topic_model']
    TEXT_SAMPLING_PROPORTION = _properties['text_sampling_proportion']
    TOPIC_WEIGHTING_METHOD = _properties['topic_weighting_method']
    LDA_BETA_COMPARISON_OPERATOR = _properties['lda_beta_comparison_operator']
    BOW_TYPE = _properties['bow_type']
    LEMMATIZE = _properties['lemmatize']
    MIN_DICTIONARY_WORD_COUNT = _properties['min_dictionary_word_count']
    MAX_DICTIONARY_WORD_COUNT = _properties['max_dictionary_word_count']
    DOCUMENT_LEVEL = _properties['document_level']
    SOLVER = _properties['solver']
    FASTFM_METHOD = _properties['fastfm_method']
    EVALUATION_METRIC = _properties['evaluation_metric']
    RESAMPLER = _properties['resampler']
    DOCUMENT_CLASSIFIER = _properties['document_classifier']
    DOCUMENT_CLASSIFIER_SEED = _properties['document_classifier_seed']
    TEST_CONTEXT_REVIEWS_ONLY = _properties['test_context_reviews_only']
    USE_NO_CONTEXT_TOPICS_SUM = _properties['use_no_context_topics_sum']
    FM_USE_BIAS = int(_properties['fm_use_bias'])
    FM_USE_1WAY_INTERACTIONS = int(_properties['fm_use_1way_interactions'])
    FM_ITERATIONS = _properties['fm_iterations']
    FM_INIT_STDEV = _properties['fm_init_stdev']
    MAX_SAMPLE_TEST_SET = _properties['max_sample_test_set']
    NESTED_CROSS_VALIDATION_CYCLE = _properties['nested_cross_validation_cycle']
    CROSS_VALIDATION_STRATEGY = _properties['cross_validation_strategy']

    # Main Files
    CACHE_FOLDER = DATASET_FOLDER + 'cache_context/'
    # RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
    #                ITEM_TYPE + 's_shuffled_tagged.json'
    RECORDS_FILE =\
        DATASET_FOLDER + 'yelp_training_set_review_' + ITEM_TYPE + 's.json'
    CLASSIFIED_RECORDS_FILE = DATASET_FOLDER + 'classified_' + ITEM_TYPE +\
        '_reviews' + ('' if DOCUMENT_LEVEL == 'review' else '_sentences') +\
        '.json'
    PROCESSED_RECORDS_FILE =\
        CACHE_FOLDER + ITEM_TYPE + '_processed_reviews' +\
        ('' if BOW_TYPE is None else '_' + BOW_TYPE) +\
        '_' + str(DOCUMENT_LEVEL) + '.json'
    FULL_PROCESSED_RECORDS_FILE =\
        CACHE_FOLDER + ITEM_TYPE + '_full_processed_reviews' + \
        ('' if BOW_TYPE is None else '_' + BOW_TYPE) +\
        '_' + str(DOCUMENT_LEVEL) + '.json'
    DICTIONARY_FILE = CACHE_FOLDER + ITEM_TYPE + '_dictionary' + \
        ('' if BOW_TYPE is None else '_' + BOW_TYPE) +\
        '_' + str(DOCUMENT_LEVEL) + '.pkl'
    REVIEWS_FILE = DATASET_FOLDER + 'reviews_' + ITEM_TYPE + '_shuffled.pkl'
    CSV_RESULTS_FILE = DATASET_FOLDER + \
        ITEM_TYPE + '_results-1iter.csv'
    JSON_RESULTS_FILE = DATASET_FOLDER + \
        ITEM_TYPE + '_results-1iter.json'
    GIT_REVISION_HASH = strip(subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=CODE_FOLDER))
    _properties['git_revision_hash'] = GIT_REVISION_HASH
    OS_NAME = platform.system() + ' ' + platform.release()
    _properties['os_name'] = OS_NAME
    # Cache files
    USER_ITEM_MAP_FILE = CACHE_FOLDER +\
        ITEM_TYPE + '_' + 'user_item_map.pkl'
    TOPIC_MODEL_FILE = CACHE_FOLDER + 'topic_model_' +\
        ITEM_TYPE + '.pkl'

    @staticmethod
    def update_properties(new_properties):
        Constants._properties.update(new_properties)

        Constants.ITEM_TYPE = Constants._properties['business_type']
        Constants.REVIEW_TYPE = Constants._properties['review_type']
        Constants.TOPN_N = Constants._properties['topn_n']
        Constants.TOPN_NUM_ITEMS = Constants._properties['topn_num_items']
        Constants.RANDOM_SEED = Constants._properties['random_seed']
        Constants.NUMPY_RANDOM_SEED = Constants._properties['numpy_random_seed']
        Constants.NUM_CYCLES = Constants._properties['num_cycles']
        Constants.LDA_ALPHA = Constants._properties['lda_alpha']
        Constants.LDA_BETA = Constants._properties['lda_beta']
        Constants.LDA_EPSILON = Constants._properties['lda_epsilon']
        Constants.LDA_NUM_TOPICS = Constants._properties['lda_num_topics']
        Constants.LDA_MODEL_PASSES = Constants._properties['lda_model_passes']
        Constants.LDA_MODEL_ITERATIONS =\
            Constants._properties['lda_model_iterations']
        Constants.LDA_MULTICORE = Constants._properties['lda_multicore']
        Constants.LIBFM_SEED = Constants._properties['libfm_seed']
        Constants.FM_NUM_FACTORS = Constants._properties['fm_num_factors']
        Constants.CROSS_VALIDATION_NUM_FOLDS =\
            Constants._properties['cross_validation_num_folds']
        Constants.SHUFFLE_DATA = Constants._properties['shuffle_data']
        Constants.USE_CONTEXT = Constants._properties['use_context']
        Constants.NUM_CORES = Constants._properties['num_cores']
        Constants.CACHE_TOPIC_MODEL = Constants._properties['cache_topic_model']
        Constants.TEXT_SAMPLING_PROPORTION =\
            Constants._properties['text_sampling_proportion']
        Constants.TOPIC_WEIGHTING_METHOD =\
            Constants._properties['topic_weighting_method']
        Constants.LDA_BETA_COMPARISON_OPERATOR =\
            Constants._properties['lda_beta_comparison_operator']
        Constants.BOW_TYPE = Constants._properties['bow_type']
        Constants.LEMMATIZE = Constants._properties['lemmatize']
        Constants.MIN_DICTIONARY_WORD_COUNT =\
            Constants._properties['min_dictionary_word_count']
        Constants.MIN_DICTIONARY_WORD_COUNT =\
            Constants._properties['max_dictionary_word_count']
        Constants.DOCUMENT_LEVEL = Constants._properties['document_level']
        Constants.SOLVER = Constants._properties['solver']
        Constants.FASTFM_METHOD = Constants._properties['fastfm_method']
        Constants.EVALUATION_METRIC = Constants._properties['evaluation_metric']
        Constants.RESAMPLER = Constants._properties['resampler']
        Constants.DOCUMENT_CLASSIFIER =\
            Constants._properties['document_classifier']
        Constants.DOCUMENT_CLASSIFIER_SEED =\
            Constants._properties['document_classifier_seed']
        Constants.TEST_CONTEXT_REVIEWS_ONLY = \
            Constants._properties['test_context_reviews_only']
        Constants.USE_NO_CONTEXT_TOPICS_SUM = \
            Constants._properties['use_no_context_topics_sum']
        Constants.FM_USE_BIAS = \
            int(Constants._properties['fm_use_bias'])
        Constants.FM_USE_1WAY_INTERACTIONS = \
            int(Constants._properties['fm_use_1way_interactions'])
        Constants.FM_ITERATIONS = Constants._properties['fm_iterations']
        Constants.FM_INIT_STDEV = Constants._properties['fm_init_stdev']
        Constants.MAX_SAMPLE_TEST_SET =\
            Constants._properties['max_sample_test_set']
        Constants.NESTED_CROSS_VALIDATION_CYCLE = \
            Constants._properties['nested_cross_validation_cycle']
        Constants.CROSS_VALIDATION_STRATEGY = \
            Constants._properties['cross_validation_strategy']

        # Main Files
        Constants.CACHE_FOLDER = Constants.DATASET_FOLDER + 'cache_context/'
        Constants.RECORDS_FILE =\
            Constants.DATASET_FOLDER + 'yelp_training_set_review_' +\
            Constants.ITEM_TYPE + 's_shuffled_tagged.json'
        Constants.CSV_RESULTS_FILE = Constants.DATASET_FOLDER + \
            Constants.ITEM_TYPE + '_results.csv'
        Constants.JSON_RESULTS_FILE = Constants.DATASET_FOLDER + \
            Constants.ITEM_TYPE + '_results.json'
        Constants.GIT_REVISION_HASH = strip(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=CODE_FOLDER))
        Constants._properties['git_revision_hash'] = Constants.GIT_REVISION_HASH
        Constants.OS_NAME = platform.system() + ' ' + platform.release()
        Constants._properties['os_name'] = Constants.OS_NAME
        # Cache files
        Constants.USER_ITEM_MAP_FILE = Constants.CACHE_FOLDER +\
            Constants.ITEM_TYPE + '_' + 'user_item_map.pkl'
        Constants.TOPIC_MODEL_FILE = Constants.CACHE_FOLDER + 'topic_model_' +\
            Constants.ITEM_TYPE + '.pkl'
