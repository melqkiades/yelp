import os
from subprocess import call

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

    for fold in range(Constants.CROSS_VALIDATION_NUM_FOLDS):

        ratings_fold_folder = LIBFM_RATINGS_FOLD_FOLDER % fold
        train_file = ratings_fold_folder + 'libfm_train.libfm'
        test_file = ratings_fold_folder + 'libfm_test.libfm'
        predictions_file = ratings_fold_folder + 'libfm_predictions.txt'
        log_file = ratings_fold_folder + 'libfm_log.txt'
        save_file = ratings_fold_folder + 'libfm_model.txt'

        if not os.path.exists(ratings_fold_folder):
            os.makedirs(ratings_fold_folder)

        run_libfm(train_file, test_file, predictions_file, log_file, save_file)

main()
