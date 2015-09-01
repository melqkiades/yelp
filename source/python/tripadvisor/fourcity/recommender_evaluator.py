import time

from etl import ETLUtils
from evaluation.mean_absolute_error import MeanAbsoluteError
from evaluation.root_mean_square_error import RootMeanSquareError
from recommenders.similarity.single_similarity_matrix_builder import \
    SingleSimilarityMatrixBuilder


__author__ = 'fpena'


def predict_rating_list(predictor, reviews):
    """
    For each one of the reviews this method predicts the rating for the
    user and item contained in the review and also returns the error
    between the predicted rating and the actual rating the user gave to the
    item

    :param predictor: the object used to predict the rating that will be given
     by a the user to the item contained in each review
    :param reviews: a list of reviews (the test data)
    :return: a tuple with a list of the predicted ratings and the list of
    errors for those predictions
    """
    predicted_ratings = []
    errors = []
    num_unknown_ratings = 0.

    for review in reviews:

        user_id = review['user_id']
        item_id = review['offering_id']

        if not predictor.has_context:
            predicted_rating = predictor.predict_rating(user_id, item_id)
        else:
            text_review = review['text']
            predicted_rating =\
                predictor.predict_rating(user_id, item_id, text_review)
        actual_rating = review['overall_rating']

        # print(user_id, item_id, predicted_rating)

        error = None

        # print('actual rating', actual_rating, 'Predicted rating', predicted_rating)

        if predicted_rating is not None:
            error = abs(predicted_rating - actual_rating)
        else:
            num_unknown_ratings += 1

        predicted_ratings.append(predicted_rating)
        errors.append(error)

    return predicted_ratings, errors, num_unknown_ratings


def perform_cross_validation(records, recommender, num_folds, cache_reviews=None):

    start_time = time.time()
    split = 1 - (1/float(num_folds))
    total_mean_absolute_error = 0.
    total_mean_square_error = 0.
    total_coverage = 0.
    num_cycles = 0

    for i in range(0, num_folds):
        print('\n\nNum cycles: %d' % i)
        start = float(i) / num_folds
        train_records, test_records = ETLUtils.split_train_test(
            records, split=split, shuffle_data=False, start=start)
        if cache_reviews:
            train_reviews, test_reviews = ETLUtils.split_train_test(
                cache_reviews, split=split, shuffle_data=False, start=start)
            recommender.reviews = train_reviews
        recommender.load(train_records)
        _, errors, num_unknown_ratings = predict_rating_list(recommender, test_records)
        mean_absolute_error = MeanAbsoluteError.compute_list(errors)
        root_mean_square_error = RootMeanSquareError.compute_list(errors)
        num_samples = len(test_records)
        coverage = float((num_samples - num_unknown_ratings) / num_samples)
        # print('Total length:', len(test))
        # print('Unknown ratings:', num_unknown_ratings)
        # print('Coverage:', coverage)

        if mean_absolute_error is not None:
            total_mean_absolute_error += mean_absolute_error
            total_mean_square_error += root_mean_square_error
            total_coverage += coverage
            num_cycles += 1
        else:
            print('Mean absolute error is None!!!')


    final_mean_absolute_error = total_mean_absolute_error / num_cycles
    final_root_squared_error = total_mean_square_error / num_cycles
    final_coverage = total_coverage / num_cycles
    execution_time = time.time() - start_time

    print('Final mean absolute error: %f' % final_mean_absolute_error)
    print('Final root mean square error: %f' % final_root_squared_error)
    print('Final coverage: %f' % final_coverage)
    print("--- %s seconds ---" % execution_time)

    result = {
        'MAE': final_mean_absolute_error,
        'RMSE': final_root_squared_error,
        'Coverage': final_coverage,
        'Execution time': execution_time
    }

    return result


def evaluate_recommender_similarity_metrics(reviews, recommender):

    headers = [
        'Algorithm',
        'Multi-cluster',
        'Similarity algorithm',
        'Similarity metric',
        'Num neighbors',
        'Dataset',
        'MAE',
        'RMSE',
        'Top N',
        'Coverage',
        'Execution time',
        'Cross validation',
        'Machine'
    ]
    similarity_metrics = ['euclidean']  # , 'cosine', 'chebyshev', 'manhattan', 'pearson']
    similarity_algorithms = [
        SingleSimilarityMatrixBuilder('euclidean'),
        # AverageSimilarityMatrixBuilder('euclidean'),
        # MultiSimilarityMatrixBuilder('euclidean'),
    ]
    ranges = [
        # [(-1.001, -0.999), (0.999, 1.001)],
        # [(-1.01, -0.99), (0.99, 1.01)],
        # [(-1.05, -0.95), (0.95, 1.05)],
        # [(-1.1, -0.9), (0.9, 1.1)],
        # [(-1.2, -0.8), (0.8, 1.2)],
        # [(-1.3, -0.7), (0.7, 1.3)],
        # [(-1.5, -0.5), (0.5, 1.5)],
        # [(-1.7, -0.3), (0.3, 1.7)],
        # [(-1.9, -0.1), (0.1, 1.9)],
        None
    ]
    num_neighbors_list = [None]  # [None, 1, 3, 5, 10, 20, 30, 40]
    num_folds = 5
    results = []

    for similarity_algorithm in similarity_algorithms:

        for num_neighbors in num_neighbors_list:

            for similarity_metric in similarity_metrics:

                for cluster_range in ranges:

                    recommender._similarity_matrix_builder = similarity_algorithm
                    recommender._similarity_matrix_builder._similarity_metric = similarity_metric
                    recommender._significant_criteria_ranges = cluster_range
                    recommender._num_neighbors = num_neighbors

                    print(
                        recommender.name, recommender._significant_criteria_ranges,
                        recommender._similarity_matrix_builder._name,
                        recommender._similarity_matrix_builder._similarity_metric,
                        recommender._num_neighbors
                    )

                    result = perform_cross_validation(reviews, recommender, num_folds)
                    # result = precision_in_top_n.calculate_top_n_precision(reviews, recommender, 5, 4.0, 5)

                    # result['Top N'] = precision_in_top_n.calculate_top_n_precision(reviews, recommender, 5, 4.0, 5)['Top N']
                    result['Algorithm'] = recommender.name
                    result['Multi-cluster'] = recommender._significant_criteria_ranges
                    result['Similarity algorithm'] = recommender._similarity_matrix_builder._name
                    result['Similarity metric'] = recommender._similarity_matrix_builder._similarity_metric
                    result['Cross validation'] = 'Folds=' + str(num_folds) + ', Iterations = ' + str(num_folds)
                    result['Num neighbors'] = recommender._num_neighbors
                    result['Dataset'] = 'Four City'
                    result['Machine'] = 'Mac'
                    results.append(result)

    file_name = '/Users/fpena/tmp/rs-test/test-delete-' + recommender.name + '.csv'
    ETLUtils.save_csv_file(file_name, results, headers)


def evaluate_recommenders(reviews, recommender_list):

    for recommender in recommender_list:
        evaluate_recommender_similarity_metrics(reviews, recommender)


