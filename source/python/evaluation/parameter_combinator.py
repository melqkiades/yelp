import itertools

from utils.constants import Constants


def combine_parameters(
        business_type,
        review_type,
        num_cycles,
        topn_n,
        topn_num_items,
        lda_alpha_list,
        lda_beta_list,
        lda_epsilon_list,
        lda_num_topics_list,
        lda_model_passes_list,
        lda_model_iterations_list,
        lda_multicore_list,
        fm_num_factors_list,
        cross_validation_num_folds,
        use_context,
        random_seed,
        numpy_random_seed,
        libfm_seed,
        shuffle_data,
        num_cores
        ):

    combined_parameters = []

    for lda_alpha,\
        lda_beta,\
        lda_epsilon,\
        lda_num_topics,\
        lda_model_passes,\
        lda_model_iterations,\
        lda_multicore,\
        fm_num_factors\
        in itertools.product(
            lda_alpha_list,
            lda_beta_list,
            lda_epsilon_list,
            lda_num_topics_list,
            lda_model_passes_list,
            lda_model_iterations_list,
            lda_multicore_list,
            fm_num_factors_list
            ):

        parameters = {
            Constants.BUSINESS_TYPE_FIELD: business_type,
            Constants.FM_REVIEW_TYPE_FIELD: review_type,
            Constants.NUM_CYCLES_FIELD: num_cycles,
            Constants.TOPN_N_FIELD: topn_n,
            Constants.TOPN_NUM_ITEMS_FIELD: topn_num_items,
            Constants.CONTEXT_EXTRACTOR_ALPHA_FIELD: lda_alpha,
            Constants.CONTEXT_EXTRACTOR_BETA_FIELD: lda_beta,
            Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD: lda_epsilon,
            Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: lda_num_topics,
            Constants.TOPIC_MODEL_PASSES_FIELD: lda_model_passes,
            Constants.TOPIC_MODEL_ITERATIONS_FIELD: lda_model_iterations,
            Constants.LDA_MULTICORE: lda_multicore,
            Constants.FM_NUM_FACTORS_FIELD: fm_num_factors,
            Constants.CROSS_VALIDATION_NUM_FOLDS_FIELD: cross_validation_num_folds,
            Constants.USE_CONTEXT_FIELD: use_context,
            Constants.RANDOM_SEED_FIELD: random_seed,
            Constants.NUMPY_RANDOM_SEED_FIELD: numpy_random_seed,
            Constants.LIBFM_SEED_FIELD: libfm_seed,
            Constants.SHUFFLE_DATA_FIELD: shuffle_data,
            Constants.NUM_CORES_FIELD: num_cores
        }

        # We remove parameters with None value from the dictionary
        parameters = {k: v for k, v in parameters.items() if v is not None}
        print('parameters', parameters)

        combined_parameters.append(parameters)

    return combined_parameters


def get_combined_parameters():
    business_type = 'yelp_hotel'
    topn_num_items = 270
    # business_type = 'yelp_restaurant'
    # topn_num_items = 1000
    review_type = None
    num_cycles = None
    topn_n = None
    lda_alpha_list = [None]
    lda_beta_list = [None]
    lda_epsilon_list = [None]
    lda_num_topics_list = [None]
    lda_model_passes_list = [None]
    lda_model_iterations_list = [None]
    lda_multicore_list = [None]
    fm_num_factors_list = [1, 2, 4, 8, 16, 32]
    cross_validation_num_folds = None
    use_context = None
    random_seed = None
    numpy_random_seed = None
    libfm_seed = None
    shuffle_data = None
    num_cores = None

    combined_parameters = combine_parameters(
        business_type,
        review_type,
        num_cycles,
        topn_n,
        topn_num_items,
        lda_alpha_list,
        lda_beta_list,
        lda_epsilon_list,
        lda_num_topics_list,
        lda_model_passes_list,
        lda_model_iterations_list,
        lda_multicore_list,
        fm_num_factors_list,
        cross_validation_num_folds,
        use_context,
        random_seed,
        numpy_random_seed,
        libfm_seed,
        shuffle_data,
        num_cores
    )

    return combined_parameters
