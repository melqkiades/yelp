import platform
from string import strip

import yaml
import subprocess

__author__ = 'fpena'


CODE_FOLDER = '/Users/fpena/UCC/Thesis/projects/yelp/source/python/'
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
    DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/stuff/'
    LIBFM_FOLDER = '/Users/fpena/tmp/libfm-master/bin/'
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
    MAX_SENTENCES = _properties['max_sentences']
    SOLVER = _properties['solver']
    FASTFM_METHOD = _properties['fastfm_method']

    # Main Files
    CACHE_FOLDER = DATASET_FOLDER + 'cache_context/'
    # RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
    #                ITEM_TYPE + 's_shuffled_tagged.json'
    RECORDS_FILE =\
        DATASET_FOLDER + 'yelp_training_set_review_' + ITEM_TYPE + 's.json'
    PROCESSED_RECORDS_FILE =\
        CACHE_FOLDER + ITEM_TYPE + '_processed_reviews' +\
        ('' if BOW_TYPE is None else '_' + BOW_TYPE) +\
        ('' if MAX_SENTENCES is None else '_sentences') + '.json'
    FULL_PROCESSED_RECORDS_FILE =\
        CACHE_FOLDER + ITEM_TYPE + '_full_processed_reviews' + \
        ('' if BOW_TYPE is None else '_' + BOW_TYPE) +\
        ('' if MAX_SENTENCES is None else '_sentences') + '.json'
    DICTIONARY_FILE = CACHE_FOLDER + ITEM_TYPE + '_dictionary' + \
        ('' if BOW_TYPE is None else '_' + BOW_TYPE) +\
        ('' if MAX_SENTENCES is None else '_sentences') + '.pkl'
    REVIEWS_FILE = DATASET_FOLDER + 'reviews_' + ITEM_TYPE + '_shuffled.pkl'
    CSV_RESULTS_FILE = DATASET_FOLDER + \
        ITEM_TYPE + '_results.csv'
    JSON_RESULTS_FILE = DATASET_FOLDER + \
        ITEM_TYPE + '_results.json'
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
        Constants.MAX_SENTENCES = Constants._properties['max_sentences']
        Constants.SOLVER = Constants._properties['solver']
        Constants.FASTFM_METHOD = Constants._properties['fastfm_method']

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
