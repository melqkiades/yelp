import cPickle as pickle
import time
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
from recommenders.context.neighbourhood.context_neighbourhood_calculator import \
    ContextNeighbourhoodCalculator
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

    return recommender_info


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


def run_rmse_test(records_file, recommenders, binary_reviews_file):

    log = "Start time: " + time.strftime("%H:%M:%S")

    records = load_records(records_file)
    # records = extractor.remove_users_with_low_reviews(records, 2)
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

    for recommender in recommenders:
        log += "\n" + get_knn_recommender_info(recommender) + "\n"
        results = recommender_evaluator.perform_cross_validation(
            records, recommender, num_folds, binary_reviews)
        log += "\n\tMAE: " + str(results['MAE'])
        log += "\n\tRMSE: " + str(results['RMSE'])
        log += "\n\tCoverage: " + str(results['Coverage'])
        log += "\n\tExecution time: " + str(results['Execution time'])

    log += "\nFinish time: " + time.strftime("%H:%M:%S")
    print(log)
    return records


def run_top_n_test(records_file, recommenders, binary_reviews_file):

    log = "Start time: " + time.strftime("%H:%M:%S")

    records = load_records(records_file)
    # records = extractor.remove_users_with_low_reviews(records, 2)
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

    for recommender in recommenders:
        log += "\n" + get_knn_recommender_info(recommender) + "\n"
        results = precision_in_top_n.calculate_recall_in_top_n(
            records, recommender, top_n, num_folds, min_like_score,
            binary_reviews)
        log += "\n\tTop N: " + str(results['Top N'])
        log += "\n\tCoverage: " + str(results['Coverage'])
        log += "\n\tExecution time: " + str(results['Execution time'])

    log += "\nFinish time: " + time.strftime("%H:%M:%S")
    print(log)


my_records_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_hotels_shuffled.json"
my_binary_reviews_file = '/Users/fpena/tmp/reviews_hotel_shuffled.pkl'
# my_binary_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/reviews_context_hotel_2.pkl'


nc = ContextNeighbourhoodCalculator()
ncc = NeighbourContributionCalculator()
ubc = UserBaselineCalculator()
usc = PBCSimilarityCalculator()
# cosine_usc = CBCSimilarityCalculator()
snc = SimpleNeighbourhoodCalculator()
cosine_usc = CosineSimilarityCalculator()
pearson_usc = PearsonSimilarityCalculator()
subc = SimpleUserBaselineCalculator()
num_topics = 150
num_neighbours = None

numpy.random.seed(0)
contextual_knn1 = ContextualKNN(num_topics, snc, ncc, subc, pearson_usc, has_context=False)
contextual_knn1.num_neighbours = num_neighbours
contextual_knn2 = ContextualKNN(num_topics, nc, ncc, ubc, usc, has_context=True)
contextual_knn2.num_neighbours = num_neighbours
# get_knn_recommender_info(contextual_knn1)
run_rmse_test(my_records_file, [contextual_knn1], my_binary_reviews_file)
run_top_n_test(my_records_file, [contextual_knn1], my_binary_reviews_file)

