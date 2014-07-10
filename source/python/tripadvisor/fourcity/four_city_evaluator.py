import time
from etl import ETLUtils
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity import fourcity_clusterer
from tripadvisor.fourcity.clu_cf_euc import CluCFEuc

__author__ = 'fpena'


def calculate_mean_average_error(errors):
    """
    Calculates the mean average error for the predicted rating

    :param reviews: the list of all reviews
    :param user_cluster_dictionary: a dictionary where all the keys are the
    cluster names and the values for those keys are list of users that belong to
    that cluster
    :return: the mean average error after predicting all the overall ratings
    """
    num_ratings = 0.
    total_error = 0.

    for error in errors:
        if error is not None:
            total_error += error
            num_ratings += 1

    mean_absolute_error = total_error / num_ratings
    return mean_absolute_error


def calculate_root_mean_square_error(errors):
    """
    Calculates the mean average error for the predicted rating

    :param reviews: the list of all reviews
    :param user_cluster_dictionary: a dictionary where all the keys are the
    cluster names and the values for those keys are list of users that belong to
    that cluster
    :return: the mean average error after predicting all the overall ratings
    """
    num_ratings = 0.
    total_error = 0.

    for error in errors:
        if error is not None:
            total_error += error ** 2
            num_ratings += 1

    root_mean_square_error = (total_error / num_ratings) ** 0.5
    return root_mean_square_error


def perform_clu_cf_euc_cross_validation():

    # reviews = extractor.pre_process_reviews()
    reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
    num_iterations = 10
    total_mean_absolute_error = 0.
    total_mean_square_error = 0.

    for i in xrange(0, num_iterations):
        train, test = ETLUtils.split_train_test(reviews)
        clusterer = CluCFEuc(train)
        _, errors = clusterer.predict_ratings_list(test)
        mean_absolute_error = calculate_mean_average_error(errors)
        print('Mean Absolute error: %f' % mean_absolute_error)
        root_mean_square_error = calculate_root_mean_square_error(errors)
        print('Root mean square error: %f' % root_mean_square_error)
        total_mean_absolute_error += mean_absolute_error
        total_mean_square_error += root_mean_square_error

    print('Final mean absolute error: %f' % (total_mean_absolute_error / num_iterations))
    print('Final root mean square error: %f' % (total_mean_square_error / num_iterations))


def perform_clu_cf_euc_top_n_validation():
    reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
    clusterer = CluCFEuc(reviews)
    users = extractor.get_groupby_list(reviews, 'user_id')
    clusterer.calculate_recall(users, 10)


def perform_clu_overall_cross_validation():

    # reviews = extractor.pre_process_reviews()
    reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')

    for i in xrange(0, 5):
        train, test = ETLUtils.split_train_test(reviews)
        user_cluster_dictionary = fourcity_clusterer.build_user_clusters(train)
        _, errors = fourcity_clusterer.clu_overall_list(test, user_cluster_dictionary)
        mean_absolute_error = calculate_mean_average_error(errors)
        print('Mean Absolute error: %f' % mean_absolute_error)
        root_mean_square_error = calculate_root_mean_square_error(errors)
        print('Root mean square error: %f' % root_mean_square_error)

start_time = time.time()
# main()
perform_clu_cf_euc_cross_validation()
# perform_clu_cf_euc_top_n_validation()
# perform_clu_overall_cross_validation()
end_time = time.time() - start_time
print("--- %s seconds ---" % end_time)
