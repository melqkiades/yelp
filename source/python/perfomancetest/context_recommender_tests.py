import cPickle as pickle
import copy
import time
import itertools
import numpy
from etl import ETLUtils
from evaluation import precision_in_top_n
from recommenders.context.baseline.simple_user_baseline_calculator import \
    SimpleUserBaselineCalculator
from recommenders.context.baseline.user_baseline_calculator import \
    UserBaselineCalculator
from recommenders.context.contextual_knn import ContextualKNN
from recommenders.context.neighbour_contribution.neighbour_contribution_calculator import \
    NeighbourContributionCalculator
from recommenders.context.neighbour_contribution.context_nc_calculator import \
    ContextNCCalculator
from recommenders.context.neighbourhood.context_neighbourhood_calculator import \
    ContextNeighbourhoodCalculator
from recommenders.context.neighbourhood.context_hybrid_neighbourhood_calculator import \
    ContextHybridNeighbourhoodCalculator
from recommenders.context.neighbourhood.simple_neighbourhood_calculator import \
    SimpleNeighbourhoodCalculator
from recommenders.context.similarity.cbc_similarity_calculator import \
    CBCSimilarityCalculator
from recommenders.context.similarity.cosine_similarity_calculator import \
    CosineSimilarityCalculator
from recommenders.context.similarity.pbc_similarity_calculator import \
    PBCSimilarityCalculator
from recommenders.context.similarity.pearson_similarity_calculator import \
    PearsonSimilarityCalculator
from tripadvisor.fourcity import recommender_evaluator
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


def get_knn_recommender_info(recommender):

    recommender_name = recommender.__class__.__name__
    nc_name = recommender.neighbourhood_calculator.__class__.__name__
    ncc_name = recommender.neighbour_contribution_calculator.__class__.__name__
    ubc_name = recommender.user_baseline_calculator.__class__.__name__
    usc_name = recommender.user_similarity_calculator.__class__.__name__

    recommender_info = recommender_name
    recommender_info += "\n\tNeighbourhood calculator: " + nc_name
    recommender_info += "\n\tNeighbour contribution calculator: " + ncc_name
    recommender_info += "\n\tUser baseline calculator: " + ubc_name
    recommender_info += "\n\tUser similarity calculator: " + usc_name
    recommender_info += "\n\tHas context: " + str(recommender.has_context)
    recommender_info += "\n\tNumber of neighbours: " +\
                        str(recommender.num_neighbours)
    recommender_info += "\n\tNumber of topics: " + str(recommender.num_topics)
    recommender_info += "\n\tThreshold 1: " + str(recommender.threshold1)
    recommender_info += "\n\tThreshold 2: " + str(recommender.threshold2)
    recommender_info += "\n\tThreshold 3: " + str(recommender.threshold3)
    recommender_info += "\n\tThreshold 4: " + str(recommender.threshold4)

    recommender_info_map = {}
    recommender_info_map['name'] = recommender_name
    recommender_info_map['neighbourhood_calculator'] = nc_name
    recommender_info_map['neighbour_contribution_calculator'] = ncc_name
    recommender_info_map['user_baseline_calculator'] = ubc_name
    recommender_info_map['user_similarity_calculator'] = usc_name
    recommender_info_map['num_neighbours'] = recommender.num_neighbours
    recommender_info_map['num_topics'] = recommender.num_topics
    recommender_info_map['threshold1'] = recommender.threshold1
    recommender_info_map['threshold2'] = recommender.threshold2
    recommender_info_map['threshold3'] = recommender.threshold3
    recommender_info_map['threshold4'] = recommender.threshold4

    return recommender_info, recommender_info_map


def load_records(json_file):
    records = ETLUtils.load_json_file(json_file)
    fields = ['user_id', 'business_id', 'stars', 'text']
    records = ETLUtils.select_fields(fields, records)

    # We rename the 'stars' field to 'overall_rating' to take advantage of the
    # function extractor.get_user_average_overall_rating
    for record in records:
        record['overall_rating'] = record.pop('stars')
        record['offering_id'] = record.pop('business_id')

    return records


def run_rmse_test(
        records_file, recommenders, binary_reviews_file, reviews_type=None):

    log = "Start time: " + time.strftime("%H:%M:%S")

    records = load_records(records_file)
    records = extractor.remove_users_with_low_reviews(records, 2)
    with open(binary_reviews_file, 'rb') as read_file:
        binary_reviews = pickle.load(read_file)

    if len(records) != len(binary_reviews):
        raise ValueError("The records and reviews should have the same length")

    dataset_info = "Dataset: " + records_file.split('/')[-1]
    dataset_info += "\n\tCache reviews: " + binary_reviews_file.split('/')[-1]
    dataset_info += "\n\tNumber of records: " + str(len(records))
    num_folds = 5
    log += "\n" + dataset_info
    log += "\nCross validation folds: " + str(num_folds)

    results_list = []
    count = 0
    print('Total recommenders: %d' % (len(recommenders)))

    for recommender in recommenders:

        print('\n**************\n%d/%d\n**************' %
              (count, len(recommenders)))
        recommender_info = get_knn_recommender_info(recommender)
        log += "\n" + recommender_info[0] + "\n"
        results = recommender_evaluator.perform_cross_validation(
            records, recommender, num_folds, binary_reviews, reviews_type)
        log += "\n\tMAE: " + str(results['MAE'])
        log += "\n\tRMSE: " + str(results['RMSE'])
        log += "\n\tCoverage: " + str(results['Coverage'])
        log += "\n\tExecution time: " + str(results['Execution time'])

        result_log = recommender_info[1]
        result_log['dataset'] = records_file.split('/')[-1]
        result_log['cache_reviews'] = binary_reviews_file.split('/')[-1]
        result_log['num_records'] = len(records)
        result_log['reviews_type'] = reviews_type
        result_log['cross_validation_folds'] = num_folds
        result_log['MAE'] = results['MAE']
        result_log['RMSE'] = results['RMSE']
        result_log['coverage'] = results['Coverage']
        result_log['time'] = results['Execution time']

        results_list.append(result_log)
        remaining_time = results['Execution time'] * (len(recommenders) - count)
        remaining_time /= 3600
        print('Estimated remaining time: %.2f hours' % remaining_time)
        count += 1

    log += "\nFinish time: " + time.strftime("%H:%M:%S")
    print(log)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'recommender-rmse-results' + timestamp
    f = open(file_name + '.log', 'w')
    f.write(log)  # python will convert \n to os.linesep
    f.close()

    headers = [
        'dataset',
        'cache_reviews',
        'num_records',
        'reviews_type',
        'cross_validation_folds',
        'RMSE',
        'MAE',
        'coverage',
        'time',
        'name',
        'neighbourhood_calculator',
        'neighbour_contribution_calculator',
        'user_baseline_calculator',
        'user_similarity_calculator',
        'num_neighbours',
        'num_topics',
        'threshold1',
        'threshold2',
        'threshold3',
        'threshold4'
    ]

    ETLUtils.save_csv_file(file_name + '.csv', results_list, headers, '\t')


def run_top_n_test(
        records_file, recommenders, binary_reviews_file, reviews_type=None):

    log = "Start time: " + time.strftime("%H:%M:%S")

    records = load_records(records_file)
    records = extractor.remove_users_with_low_reviews(records, 2)
    with open(binary_reviews_file, 'rb') as read_file:
        binary_reviews = pickle.load(read_file)

    if len(records) != len(binary_reviews):
        raise ValueError("The records and reviews should have the same length")

    dataset_info = "Dataset: " + records_file.split('/')[-1]
    dataset_info += "\n\tChache reviews: " + binary_reviews_file.split('/')[-1]
    dataset_info += "\n\tNumber of records: " + str(len(records))
    num_folds = 5
    min_like_score = 5.0
    top_n = 10
    log += "\n" + dataset_info
    log += "\nCross validation folds: " + str(num_folds)
    log += "\nMin score to like: " + str(min_like_score)
    log += "\nTop N: " + str(top_n)

    results_list = []
    count = 0
    print('Total recommenders: %d' % (len(recommenders)))

    for recommender in recommenders:

        print('\n**************\nProgress: %d/%d\n**************' %
              (count, len(recommenders)))

        recommender_info = get_knn_recommender_info(recommender)
        log += "\n" + recommender_info[0] + "\n"
        results = precision_in_top_n.calculate_recall_in_top_n(
            records, recommender, top_n, num_folds, min_like_score,
            binary_reviews, reviews_type)
        log += "\n\tTop N: " + str(results['Top N'])
        log += "\n\tCoverage: " + str(results['Coverage'])
        log += "\n\tExecution time: " + str(results['Execution time'])

        result_log = recommender_info[1]
        result_log['dataset'] = records_file.split('/')[-1]
        result_log['cache_reviews'] = binary_reviews_file.split('/')[-1]
        result_log['num_records'] = len(records)
        result_log['reviews_type'] = reviews_type
        result_log['cross_validation_folds'] = num_folds
        result_log['min_like_score'] = min_like_score
        result_log['top_n'] = top_n
        result_log['recall'] = results['Top N']
        result_log['coverage'] = results['Coverage']
        result_log['time'] = results['Execution time']

        results_list.append(result_log)
        remaining_time = results['Execution time'] * (len(recommenders) - count)
        remaining_time /= 3600
        print('Estimated remaining time: %.2f hours' % remaining_time)
        count += 1

    log += "\nFinish time: " + time.strftime("%H:%M:%S") + "\n\n"
    print(log)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'recommender-topn-results' + timestamp
    f = open(file_name + '.log', 'w')
    f.write(log)  # python will convert \n to os.linesep
    f.close()

    headers = [
        'dataset',
        'cache_reviews',
        'num_records',
        'cross_validation_folds',
        'min_like_score',
        'top_n',
        'recall',
        'coverage',
        'time',
        'name',
        'neighbourhood_calculator',
        'neighbour_contribution_calculator',
        'user_baseline_calculator',
        'user_similarity_calculator',
        'num_neighbours',
        'num_topics',
        'threshold1',
        'threshold2',
        'threshold3',
        'threshold4'
    ]

    ETLUtils.save_csv_file(file_name + '.csv', results_list, headers, '\t')


def main():

    folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    my_records_file = folder + 'yelp_training_set_review_hotels_shuffled.json'
    # my_binary_reviews_file = folder + 'reviews_restaurant_shuffled.pkl'
    # my_binary_reviews_file = folder + 'reviews_hotel_shuffled.pkl'
    my_binary_reviews_file = folder + 'reviews_context_hotel_2.pkl'

    # nc = ContextNeighbourhoodCalculator()
    # ncc = NeighbourContributionCalculator()
    # ubc = UserBaselineCalculator()
    # usc = PBCSimilarityCalculator()
    # cosine_usc = CBCSimilarityCalculator()

    # Similarity calculators
    cosine_sc = CosineSimilarityCalculator()
    pearson_sc = PearsonSimilarityCalculator()
    pbc_sc = PBCSimilarityCalculator()
    cbu_sc = CBCSimilarityCalculator()
    similarity_calculators = [
        cosine_sc,
        pearson_sc,
        pbc_sc,
        cbu_sc
    ]
    
    # Neighbourhood calculators
    simple_nc = SimpleNeighbourhoodCalculator(copy.deepcopy(pearson_sc))
    context_nc = ContextNeighbourhoodCalculator()
    # hybrid_nc0 = ContextHybridNeighbourhoodCalculator(copy.deepcopy(pearson_sc))
    # hybrid_nc0.weight = 0.0
    # hybrid_nc02 = ContextHybridNeighbourhoodCalculator(copy.deepcopy(pearson_sc))
    # hybrid_nc02.weight = 0.2
    hybrid_nc05 = ContextHybridNeighbourhoodCalculator(copy.deepcopy(pearson_sc))
    hybrid_nc05.weight = 0.5
    # hybrid_nc08 = ContextHybridNeighbourhoodCalculator(copy.deepcopy(pearson_sc))
    # hybrid_nc08.weight = 0.8
    # hybrid_nc1 = ContextHybridNeighbourhoodCalculator(copy.deepcopy(pearson_sc))
    # hybrid_nc1.weight = 1.0
    neighbourhood_calculators = [
        simple_nc,
        context_nc,
        # hybrid_nc0,
        # hybrid_nc02,
        hybrid_nc05,
        # hybrid_nc08,
        # hybrid_nc1
    ]

    # Baseline calculators
    simple_ubc = SimpleUserBaselineCalculator()
    ubc = UserBaselineCalculator()
    baseline_calculators = [
        ubc,
        simple_ubc
    ]

    # Neighbour contribution calculators
    ncc = NeighbourContributionCalculator()
    context_ncc = ContextNCCalculator()
    neighbour_contribution_calculators = [
        ncc,
        # context_ncc
    ]

    num_topics = 150
    # num_neighbours = None

    numpy.random.seed(0)
    basic_cosine_knn = ContextualKNN(num_topics, simple_nc, ncc, simple_ubc, cosine_sc, has_context=False)
    basic_pearson_knn = ContextualKNN(num_topics, simple_nc, ncc, simple_ubc, pearson_sc, has_context=False)
    contextual_knn = ContextualKNN(num_topics, context_nc, ncc, ubc, pbc_sc, has_context=True)
    # get_knn_recommender_info(contextual_knn1)

    # ocelma_recommender = OcelmaRecommender()

    recommenders = [
        # basic_cosine_knn,
        # basic_pearson_knn,
        contextual_knn
        # ocelma_recommender
    ]

    num_neighbours_list = [None]
    # num_neighbours_list = [None, 3, 6, 10, 15, 20]
    threshold_list = [0.0, 0.5, 0.9]
    # num_topics_list = [10, 50, 150, 300, 500]
    num_topics_list = [150]

    # combined_recommenders = []
    # for recommender, num_neighbours in itertools.product(recommenders, num_neighbours_list):
    #     new_recommender = copy.deepcopy(recommender)
    #     new_recommender.num_neighbours = num_neighbours
    #     combined_recommenders.append(new_recommender)

    # threshold_list = [None]
    #
    # combined_recommenders = []
    # for recommender, threshold in itertools.product(recommenders, threshold_list):
    #     new_recommender = copy.deepcopy(recommender)
    #     new_recommender.threshold1 = threshold
    #     new_recommender.threshold2 = threshold
    #     new_recommender.threshold3 = threshold
    #     new_recommender.threshold4 = threshold
    #     combined_recommenders.append(new_recommender)


    # num_threshold_list = [0.2, 0.5, 0.7]

    combined_recommenders = combine_recommenders(
        neighbourhood_calculators,
        neighbour_contribution_calculators,
        baseline_calculators,
        similarity_calculators,
        num_neighbours_list,
        threshold_list,
        num_topics_list
    )

    run_rmse_test(my_records_file, combined_recommenders[71:], my_binary_reviews_file)
    # run_top_n_test(my_records_file, combined_recommenders, my_binary_reviews_file)

    # run_rmse_test(my_records_file, combined_recommenders[47:], my_binary_reviews_file, 'specific')
    # run_top_n_test(my_records_file, combined_recommenders, my_binary_reviews_file, 'specific')

    # run_rmse_test(my_records_file, combined_recommenders[47:], my_binary_reviews_file, 'generic')
    # run_top_n_test(my_records_file, combined_recommenders, my_binary_reviews_file, 'generic')


def combine_recommenders(
        neighbourhood_calculators,
        neighbour_contribution_calculators,
        baseline_calculators,
        similarity_calculators,
        num_neighbours_list,
        thresholds,
        num_topics_list):

    recommender = ContextualKNN(None, None, None, None, None, has_context=True)
    combined_recommenders = []

    for neighbourhood_calculator,\
        neighbour_contribution_calculator,\
        baseline_calculator,\
        similarity_calculator,\
        num_neighbours,\
        threshold,\
        num_topics_list\
        in itertools.product(
            neighbourhood_calculators,
            neighbour_contribution_calculators,
            baseline_calculators,
            similarity_calculators,
            num_neighbours_list,
            thresholds,
            num_topics_list):
        new_recommender = copy.deepcopy(recommender)
        new_recommender.neighbourhood_calculator = neighbourhood_calculator
        new_recommender.neighbour_contribution_calculator =\
            neighbour_contribution_calculator
        new_recommender.user_baseline_calculator = baseline_calculator
        new_recommender.user_similarity_calculator = similarity_calculator
        new_recommender.num_neighbours = num_neighbours
        combined_recommenders.append(new_recommender)

    return combined_recommenders


main()
