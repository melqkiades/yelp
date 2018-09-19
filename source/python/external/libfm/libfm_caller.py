import os
from subprocess import call

import time

from utils.constants import Constants


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
    print('%s: Making preidctions with LibFM' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))

    prediction_type_map = {
        'user_test': 'rating',
        'test_items': 'rating',
        'rel_plus_n': 'ranking'
    }
    prediction_type = prediction_type_map[Constants.RIVAL_EVALUATION_STRATEGY]
    use_cache = True

    for fold in range(Constants.CROSS_VALIDATION_NUM_FOLDS):

        ratings_fold_folder = Constants.RIVAL_RATINGS_FOLD_FOLDER % fold
        # ratings_fold_folder = Constants.CACHE_FOLDER + 'rival/contextaa/fold_%d/' % fold
        train_file = ratings_fold_folder + 'libfm_train.libfm'
        predictions_file = ratings_fold_folder + 'libfm_predictions_' + \
                    prediction_type + '.libfm'
        fm_num_factors = Constants.FM_NUM_FACTORS
        results_file = ratings_fold_folder + 'libfm_results_' + \
            prediction_type + '_fmfactors-' + str(fm_num_factors) + '.txt'

        if use_cache and os.path.exists(results_file):
            print("Fold %d file already exists ('%s') " % (fold, results_file))
            continue

        # predictions_file = ratings_fold_folder + 'libfm_test.libfm'
        # results_file = ratings_fold_folder + 'libfm_predictions.txt'
        log_file = ratings_fold_folder + 'libfm_log.txt'
        save_file = ratings_fold_folder + 'libfm_model.txt'

        if not os.path.exists(ratings_fold_folder):
            os.makedirs(ratings_fold_folder)

        run_libfm(train_file, predictions_file, results_file, log_file, save_file)

# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
