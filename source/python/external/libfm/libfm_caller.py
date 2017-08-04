import os
from subprocess import call

import time

from utils.constants import Constants


# LIBFM_RATINGS_FOLD_FOLDER = Constants.generate_file_name(
#         'recsys_contextual_records', '', Constants.CACHE_FOLDER + 'rival/',
#         None, None, True, True, normalize_topics=True)[:-1] + '/fold_%d/'
LIBFM_RATINGS_FOLD_FOLDER = Constants.generate_file_name(
        'recsys_formatted_context_records', '', Constants.CACHE_FOLDER + 'rival/',
        None, None, True, True, uses_carskit=False, normalize_topics=True,
        format_context=True)[:-1] + '/fold_%d/'
# LIBFM_RATINGS_FOLD_FOLDER = Constants.CACHE_FOLDER + 'rival/' +\
#     'yelp_hotel_recsys_contextual_records_lang-en_bow-NN_' \
#     'document_level-review_targettype-context_min_item_reviews-10/fold_%d/'


def run_libfm(train_file, test_file, predictions_file, log_file, save_file):
    print('%s: Run LibFM' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    libfm_command = Constants.LIBFM_FOLDER + 'libFM'

    command = [
        libfm_command,
        '-task',
        'r',
        '-method',
        Constants.FM_METHOD,
        '-regular',
        str(Constants.FM_REGULARIZATION0) + ',' +
        str(Constants.FM_REGULARIZATION1) + ',' +
        str(Constants.FM_REGULARIZATION2),
        '-learn_rate',
        str(Constants.FM_SDG_LEARN_RATE),
        '-train',
        train_file,
        '-test',
        test_file,
        '-dim',
        ','.join(map(str,
                     [Constants.FM_USE_BIAS, Constants.FM_USE_1WAY_INTERACTIONS,
                      Constants.FM_NUM_FACTORS])),
        '-init_stdev',
        str(Constants.FM_INIT_STDEV),
        '-iter',
        str(Constants.FM_ITERATIONS),
        '-out',
        predictions_file,
        '-save_model',
        save_file
    ]

    print(command)

    if Constants.LIBFM_SEED is not None:
        command.extend(['-seed', str(Constants.LIBFM_SEED)])

    f = open(log_file, "w")
    call(command, stdout=f)


def main():


    prediction_type_map = {
        'user_test': 'rating',
        'test_items': 'rating',
        'rel_plus_n': 'ranking'
    }
    prediction_type = prediction_type_map[Constants.RIVAL_EVALUATION_STRATEGY]

    for fold in range(Constants.CROSS_VALIDATION_NUM_FOLDS):

        ratings_fold_folder = LIBFM_RATINGS_FOLD_FOLDER % fold
        # ratings_fold_folder = Constants.CACHE_FOLDER + 'rival/contextaa/fold_%d/' % fold
        train_file = ratings_fold_folder + 'libfm_train.libfm'
        predictions_file = ratings_fold_folder + 'libfm_predictions_' + \
                    prediction_type + '.libfm'
        results_file = ratings_fold_folder + 'libfm_results_' + \
                           prediction_type + '.txt'
        # predictions_file = ratings_fold_folder + 'libfm_test.libfm'
        # results_file = ratings_fold_folder + 'libfm_predictions.txt'
        log_file = ratings_fold_folder + 'libfm_log.txt'
        save_file = ratings_fold_folder + 'libfm_model.txt'

        if not os.path.exists(ratings_fold_folder):
            os.makedirs(ratings_fold_folder)

        run_libfm(train_file, predictions_file, results_file, log_file, save_file)

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
