import itertools


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
        lda_multicore\
        in itertools.product(
            lda_alpha_list,
            lda_beta_list,
            lda_epsilon_list,
            lda_num_topics_list,
            lda_model_passes_list,
            lda_model_iterations_list,
            lda_multicore_list
            ):

        parameters = {
            'business_type': business_type,
            'review_type': review_type,
            'num_cycles': num_cycles,
            'topn_n': topn_n,
            'topn_num_items': topn_num_items,
            'lda_alpha': lda_alpha,
            'lda_beta': lda_beta,
            'lda_epsilon': lda_epsilon,
            'lda_num_topics': lda_num_topics,
            'lda_model_passes': lda_model_passes,
            'lda_model_iterations': lda_model_iterations,
            'lda_multicore': lda_multicore,
            'cross_validation_num_folds': cross_validation_num_folds,
            'use_context': use_context,
            'random_seed': random_seed,
            'numpy_random_seed': numpy_random_seed,
            'libfm_seed': libfm_seed,
            'shuffle_data': shuffle_data,
            'num_cores': num_cores
        }

        combined_parameters.append(parameters)

    return combined_parameters


def hotel_context_parameters():
    business_type = 'hotel'
    review_type = None
    num_cycles = 10
    topn_n = 10
    topn_num_items = 270
    lda_alpha_list = [0.005]
    lda_beta_list = [1.0, 0.5, 0.0]
    lda_epsilon_list = [0.01]
    lda_num_topics_list = [450, 150, 50]
    lda_model_passes_list = [10, 1]
    lda_model_iterations_list = [500, 50]
    lda_multicore_list = [False]
    cross_validation_num_folds = 5
    use_context = True
    random_seed = 0
    numpy_random_seed = 0
    libfm_seed = 0
    shuffle_data = True
    num_cores = None

    # business_type = 'hotel'
    # review_type = None
    # num_cycles = 2
    # topn_n = 10
    # topn_num_items = 45
    # lda_alpha_list = [0.005]
    # lda_beta_list = [1.0]
    # lda_epsilon_list = [0.01]
    # lda_num_topics_list = [50]
    # lda_model_passes_list = [1]
    # lda_model_iterations_list = [50]
    # lda_multicore_list = [False]
    # cross_validation_num_folds = 5
    # use_context = True
    # random_seed = 0
    # numpy_random_seed = 0
    # libfm_seed = 0
    # shuffle_data = True
    # num_cores = None

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
        cross_validation_num_folds,
        use_context,
        random_seed,
        numpy_random_seed,
        libfm_seed,
        shuffle_data,
        num_cores
    )

    return combined_parameters


def restaurant_context_parameters():
    business_type = 'restaurant'
    review_type = None
    num_cycles = 1
    topn_n = 10
    topn_num_items = 1000
    lda_alpha_list = [0.005]
    lda_beta_list = [1.0, 0.5, 0.0]
    lda_epsilon_list = [0.01]
    lda_num_topics_list = [450, 150, 50]
    lda_model_passes_list = [10, 1]
    lda_model_iterations_list = [500, 50]
    lda_multicore_list = [False]
    cross_validation_num_folds = 20
    use_context = True
    random_seed = 0
    numpy_random_seed = 0
    libfm_seed = 0
    shuffle_data = True
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
        cross_validation_num_folds,
        use_context,
        random_seed,
        numpy_random_seed,
        libfm_seed,
        shuffle_data,
        num_cores
    )

    return combined_parameters


def hotel_no_context_parameters():
    business_type = 'hotel'
    review_type = None
    num_cycles = 10
    topn_n = 10
    topn_num_items = 270
    lda_alpha_list = [None]
    lda_beta_list = [None]
    lda_epsilon_list = [None]
    lda_num_topics_list = [None]
    lda_model_passes_list = [None]
    lda_model_iterations_list = [None]
    lda_multicore_list = [None]
    cross_validation_num_folds = 5
    use_context = False
    random_seed = 0
    numpy_random_seed = 0
    libfm_seed = 0
    shuffle_data = True
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
        cross_validation_num_folds,
        use_context,
        random_seed,
        numpy_random_seed,
        libfm_seed,
        shuffle_data,
        num_cores
    )

    return combined_parameters


def restaurant_no_context_parameters():
    business_type = 'restaurant'
    review_type = None
    num_cycles = 1
    topn_n = 10
    topn_num_items = 1000
    lda_alpha_list = [None]
    lda_beta_list = [None]
    lda_epsilon_list = [None]
    lda_num_topics_list = [None]
    lda_model_passes_list = [None]
    lda_model_iterations_list = [None]
    lda_multicore_list = [None]
    cross_validation_num_folds = 50
    use_context = False
    random_seed = 0
    numpy_random_seed = 0
    libfm_seed = 0
    shuffle_data = True
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
        cross_validation_num_folds,
        use_context,
        random_seed,
        numpy_random_seed,
        libfm_seed,
        shuffle_data,
        num_cores
    )

    return combined_parameters
