import yaml

__author__ = 'fpena'


CODE_FOLDER = '/Users/fpena/UCC/Thesis/projects/yelp/source/python/'
PROPERTIES_FILE = CODE_FOLDER + 'properties.yaml'


def load_properties():
    with open(PROPERTIES_FILE, 'r') as f:
        return yaml.load(f)


# Please keep the constants' names in alphabetical order to avoid problems
# with the version control system (merging)

CONTEXT_TOPICS_FIELD = 'context_topics'
ITEM_ID_FIELD = 'business_id'
PREDICTED_CLASS_FIELD = 'predicted_class'
RATING_FIELD = 'stars'
REVIEW_ID_FIELD = 'review_id'
TEXT_FIELD = 'text'
TOPICS_FIELD = 'topics'
USER_ID_FIELD = 'user_id'

# Folders
DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/stuff/'
LIBFM_FOLDER = '/Users/fpena/tmp/libfm-master/bin/'
GENERATED_FOLDER = DATASET_FOLDER + 'generated_context/'

_properties = load_properties()
ITEM_TYPE = _properties['business_type']
SPLIT_PERCENTAGE = _properties['split_percentage']
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

# Main Files
CACHE_FOLDER = DATASET_FOLDER + 'cache_context/'
RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
               ITEM_TYPE + 's_shuffled_tagged.json'

# Cache files
USER_ITEM_MAP_FILE = CACHE_FOLDER + ITEM_TYPE + '_' + 'user_item_map.pkl'
TOPIC_MODEL_FILE = CACHE_FOLDER + 'topic_model_' + ITEM_TYPE + '.pkl'

#
# print(_properties)
# print(TOPIC_MODEL_FILE)

