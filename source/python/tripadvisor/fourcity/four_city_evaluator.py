from random import shuffle
import time

from etl import ETLUtils
from evaluation import precision_in_top_n
from evaluation.mean_absolute_error import MeanAbsoluteError
from evaluation.root_mean_square_error import RootMeanSquareError
from recommenders.adjusted_weighted_sum_recommender import \
    AdjustedWeightedSumRecommender
from recommenders.average_recommender import AverageRecommender
from recommenders.dummy_recommender import DummyRecommender
from recommenders.matrixfactorization.stochastic_gradient_descent import \
    StochasticGradientDescent
from recommenders.similarity.single_similarity_matrix_builder import \
    SingleSimilarityMatrixBuilder
from recommenders.weighted_sum_recommender import WeightedSumRecommender
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity import movielens_extractor
from etl.reviews_dataset_analyzer import ReviewsDatasetAnalyzer
from tripadvisor.fourcity.recommender_evaluator import evaluate_recommenders


__author__ = 'fpena'


start_time = time.time()
# main()
# file_path = '/Users/fpena/tmp/filtered_reviews_multi.json'
# file_path = '/Users/fpena/tmp/filtered_reviews_multi_new.json'
# file_path = '/Users/fpena/tmp/filtered_reviews_multi_non_sparse_shuffled.json'
# file_path = '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/all_reviews.json'
file_path = '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/cleaned_reviews.json'
# reviews = ETLUtils.load_json_file(file_path)
# reviews = movielens_extractor.clean_reviews(movielens_extractor.get_ml_100K_dataset())
# reviews = extractor.pre_process_reviews()
reviews = movielens_extractor.get_ml_100K_dataset()

# shuffle(reviews)
# ETLUtils.save_json_file('/Users/fpena/tmp/filtered_reviews_multi_non_sparse_shuffled.json', reviews)
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
    # AdjustedWeightedSumRecommender(SingleSimilarityMatrixBuilder('euclidean')),
    # AdjustedWeightedSumRecommender(MultiSimilarityMatrixBuilder('chebyshev')),
    # WeightedSumRecommender(SingleSimilarityMatrixBuilder('euclidean')),
    # WeightedSumRecommender(MultiSimilarityMatrixBuilder('cosine')),
    # DeltaRecommender(),
    # DeltaCFRecommender(),
    # OverallRecommender(),
    # OverallCFRecommender(),
    StochasticGradientDescent(2),
    AverageRecommender(),
    DummyRecommender(4.0)
]

# reviewsDatasetAnalyzer = ReviewsDatasetAnalyzer(reviews)
# common_item_counts = reviewsDatasetAnalyzer.count_items_in_common()
# print(common_item_counts)
# print(reviewsDatasetAnalyzer.calculate_sparsity())
# print(reviewsDatasetAnalyzer.analyze_common_items_count(common_item_counts))
# print(reviewsDatasetAnalyzer.analyze_common_items_count(common_item_counts, True))

# predict_rating()
# my_reviews = extractor.load_json_file('/Users/fpena/tmp/filtered_reviews.json')
evaluate_recommenders(reviews, my_recommender_list)
# recommender = SingleCF('pearson')
# evaluate_recommender_similarity_metrics(recommender)
# recommender = OverallCFRecommender('euclidean')
# evaluate_recommender_similarity_metrics(recommender)
# perform_clu_cf_euc_top_n_validation()
# perform_clu_overall_cross_validation()
# perform_clu_overall_whole_dataset_evaluation()
# precision_in_top_n.calculate_top_n_precision(reviews, my_recommender_list[0], 5, 4.0, 5)
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
