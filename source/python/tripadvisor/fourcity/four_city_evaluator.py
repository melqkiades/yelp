import time

from etl import ETLUtils
from evaluation.mean_absolute_error import MeanAbsoluteError
from evaluation.root_mean_square_error import RootMeanSquareError
from recommenders.multicriteria.delta_cf_recommender import DeltaCFRecommender
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity.clu_cf_euc import CluCFEuc
from recommenders.dummy_recommender import DummyRecommender


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

    for review in reviews:

        user_id = review['user_id']
        item_id = review['offering_id']
        predicted_rating = predictor.predict_rating(user_id, item_id)
        actual_rating = review['overall_rating']

        # print(user_id, item_id, predicted_rating)

        error = None

        if predicted_rating is not None and actual_rating is not None:
            error = abs(predicted_rating - actual_rating)

        predicted_ratings.append(predicted_rating)
        errors.append(error)

    return predicted_ratings, errors


def perform_clu_cf_euc_top_n_validation():
    reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
    clusterer = CluCFEuc(reviews)
    users = extractor.get_groupby_list(reviews, 'user_id')
    clusterer.calculate_recall(users, 10)


def perform_clu_overall_cross_validation():

    headers = [
        'Algorithm',
        'Multi-cluster',
        'Similarity',
        'Distance metric',
        'Dataset',
        'MAE',
        'RMSE',
        'Execution time',
        'Cross validation',
        'Machine'
    ]

    # reviews = extractor.pre_process_reviews()
    reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
    num_iterations = 5
    split = 1 - (1/float(num_iterations))
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
    results = []

    for my_range in ranges:

        start_time = time.time()
        total_mean_absolute_error = 0.
        total_mean_square_error = 0.
        num_cycles = 0
        for i in xrange(0, num_iterations):
            start = float(i) / num_iterations
            train, test = ETLUtils.split_train_test(reviews, split=split, shuffle_data=False, start=start)
            # clusterer = CluOverall(train, my_range, True)
            clusterer = CluCFEuc(train, True, my_range)
            # clusterer = DummyPredictor(train)
            # clusterer = SingleCF(train)
            _, errors = predict_rating_list(clusterer, test)
            mean_absolute_error = MeanAbsoluteError.compute_list(errors)
            # print('Mean Absolute error:', mean_absolute_error)
            root_mean_square_error = RootMeanSquareError.compute_list(errors)
            # print('Root mean square error:',  root_mean_square_error)

            if mean_absolute_error is not None:
                total_mean_absolute_error += mean_absolute_error
                total_mean_square_error += root_mean_square_error
                num_cycles += 1

        print(my_range)
        final_mean_absolute_error = total_mean_absolute_error / num_cycles
        final_root_squared_error = total_mean_square_error / num_cycles
        print('Final mean absolute error: %f' % final_mean_absolute_error)
        print('Final root mean square error: %f' % final_root_squared_error)

        execution_time = time.time() - start_time
        print("--- %s seconds ---" % execution_time)

        result = {}
        result['Algorithm'] = 'Clu_CF',
        result['Multi-cluster'] = my_range,
        result['Similarity'] = 'Pearson',
        # results['Distance metric'] = ,
        result['Dataset'] = 'Four City',
        result['MAE'] = final_mean_absolute_error,
        result['RMSE'] = final_root_squared_error,
        result['Execution time'] = execution_time,
        result['Cross validation'] = 'Folds=' + str(num_iterations) + ', Iterations = ' + str(num_iterations),
        result['Machine'] = 'Mac'

        results.append(result)

    ETLUtils.save_csv_file('/Users/fpena/tmp/test2.csv', results, headers)


def perform_clu_overall_whole_dataset_evaluation():

    # reviews = extractor.pre_process_reviews()
    reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')

    clusterer = CluCFEuc(reviews)
    # clusterer = DummyPredictor(reviews)
    _, errors = predict_rating_list(clusterer, reviews)
    mean_absolute_error = MeanAbsoluteError.compute_list(errors)
    print('Mean Absolute error: %f' % mean_absolute_error)
    root_mean_square_error = RootMeanSquareError.compute_list(errors)
    print('Root mean square error: %f' % root_mean_square_error)


def perform_cross_validation(reviews, recommender, num_folds):

    start_time = time.time()
    split = 1 - (1/float(num_folds))
    total_mean_absolute_error = 0.
    total_mean_square_error = 0.
    num_cycles = 0
    total_errors = []

    for i in xrange(0, num_folds):
        start = float(i) / num_folds
        train, test = ETLUtils.split_train_test(reviews, split=split, shuffle_data=False, start=start)
        recommender.load(train)
        _, errors = predict_rating_list(recommender, test)
        mean_absolute_error = MeanAbsoluteError.compute_list(errors)
        root_mean_square_error = RootMeanSquareError.compute_list(errors)
        total_errors += errors

        if mean_absolute_error is not None:
            total_mean_absolute_error += mean_absolute_error
            total_mean_square_error += root_mean_square_error
            num_cycles += 1

    final_mean_absolute_error = total_mean_absolute_error / num_cycles
    final_root_squared_error = total_mean_square_error / num_cycles
    execution_time = time.time() - start_time

    print('Final mean absolute error: %f' % final_mean_absolute_error)
    print('Final root mean square error: %f' % final_root_squared_error)
    print("--- %s seconds ---" % execution_time)

    result = {
        'MAE': final_mean_absolute_error,
        'RMSE': final_root_squared_error,
        'Execution time': execution_time
    }

    return result


def evaluate_recommender_similarity_metrics(reviews, recommender):

    headers = [
        'Algorithm',
        'Multi-cluster',
        'Similarity',
        'Distance metric',
        'Dataset',
        'MAE',
        'RMSE',
        'Execution time',
        'Cross validation',
        'Machine'
    ]
    similarity_metrics = ['euclidean', 'cosine', 'pearson']

    results = []

    for similarity_metric in similarity_metrics:

        recommender.similarity_metric = similarity_metric
        num_folds = 5
        result = perform_cross_validation(reviews, recommender, num_folds)

        result['Algorithm'] = recommender.name
        result['Similarity'] = recommender.similarity_metric
        result['Cross validation'] = 'Folds=' + str(num_folds) + ', Iterations = ' + str(num_folds)
        result['Dataset'] = 'Four City'
        result['Machine'] = 'Mac'
        results.append(result)

    file_name = '/Users/fpena/tmp/rs-test/test6-' + recommender.name + '.csv'
    ETLUtils.save_csv_file(file_name, results, headers)


def evaluate_recommenders(reviews, recommender_list):

    for recommender in recommender_list:
        evaluate_recommender_similarity_metrics(reviews, recommender)




start_time = time.time()
# main()
# reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
# reviews = extractor.pre_process_reviews()
# print(reviews[0])
# print(reviews[1])
# print(reviews[2])
# print(reviews[10])
# print(reviews[100])
#
# for review in reviews:
#     print(review)


my_recommender_list = [
    # SingleCF(),
    # DeltaRecommender(),
    DeltaCFRecommender(),
    # OverallRecommender(),
    # OverallCFRecommender(),
    # DummyRecommender(4.0)
]


# my_reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
# evaluate_recommenders(my_reviews, my_recommender_list)
# recommender = SingleCF('pearson')
# evaluate_recommender_similarity_metrics(recommender)
# recommender = OverallCFRecommender('euclidean')
# evaluate_recommender_similarity_metrics(recommender)
# perform_clu_cf_euc_top_n_validation()
# perform_clu_overall_cross_validation()
# perform_clu_overall_whole_dataset_evaluation()
end_time = time.time() - start_time
print("--- %s seconds ---" % end_time)

# numerator = 4.5 * 4 + 3 * 2
# denominator = ((4.5**2)+(3**2)**0.5) * ((4**2) + (2**2) ** 0.5)
# result = numerator / denominator
# print('Result:', result)
#
# ratings1 = [4.5, 3]
# ratings2 = [4, 2]
#
# print('Cosine:', 1 - spatial.distance.cosine(ratings1, ratings2))
