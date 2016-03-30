import itertools
import math
import time
import cPickle as pickle

from gensim import corpora
import numpy

from etl import ETLUtils
from evaluation import precision_in_top_n

from recommenders.context import basic_knn
# from recommenders.context.basic.basic_contextual_knn import BasicContextualKNN
from recommenders.context.basic.basic_neighbour_contribution_calculator import \
    BasicNeighbourContributionCalculator
from recommenders.context.basic.basic_neighbourhood_calculator import \
    BasicNeighbourhoodCalculator
from recommenders.context.basic.basic_user_baseline_calculator import \
    BasicUserBaselineCalculator
from recommenders.context.basic.basic_user_similarity_calculator import \
    BasicUserSimilarityCalculator
from recommenders.context.basic_knn import BasicKNN
from recommenders.context.contextual_knn import ContextualKNN
from recommenders.context.neighbour_contribution.neighbour_contribution_calculator import \
    NeighbourContributionCalculator
from recommenders.context.neighbourhood.context_hybrid_neighbourhood_calculator import \
    ContextHybridNeighbourhoodCalculator
from recommenders.context.neighbourhood.context_neighbourhood_calculator import \
    ContextNeighbourhoodCalculator
from recommenders.context.top_k_neighbourhood_calculator import \
    TopKNeighbourhoodCalculator
from recommenders.context.baseline.user_baseline_calculator import UserBaselineCalculator
from recommenders.context.similarity.pbc_similarity_calculator import \
    PBCSimilarityCalculator
from recommenders.context.similarity.cbc_similarity_calculator import \
    CBCSimilarityCalculator
from recommenders.context.neighbourhood.simple_neighbourhood_calculator import \
    SimpleNeighbourhoodCalculator
from topicmodeling.context import lda_context_utils
from topicmodeling.context.lda_based_context import LdaBasedContext
from topicmodeling.context.review import Review
from tripadvisor.fourcity import extractor
from utils import dictionary_utils
from tripadvisor.fourcity import recommender_evaluator

__author__ = 'fpena'


class ContextKnn:

    def __init__(self, num_topics, reviews=None):
        self.records = None
        self.num_neighbors = None
        self.ratings_matrix = None
        self.reviews_matrix = None
        self.context_matrix = None
        self.similarity_matrix = None
        self.user_dictionary = None
        self.user_ids = None
        self.num_topics = num_topics
        self.lda_model = None
        self.none = None
        self.reviews = reviews
        self.context_rich_topics = None
        self.has_context = True

    def load(self, records):
        self.records = records
        self.ratings_matrix = basic_knn.create_ratings_matrix(records)
        self.reviews_matrix = create_reviews_matrix(records)
        self.user_dictionary = extractor.initialize_users(self.records, False)
        self.user_ids = extractor.get_groupby_list(self.records, 'user_id')

        lda_based_context = LdaBasedContext(self.records, self.reviews)

        # self.lda_model =\
        #     lda_context_utils.discover_topics(text_reviews, self.num_topics)
        # if self.reviews:
        #     lda_based_context = LdaBasedContext()
        #     lda_based_context.reviews = self.reviews
        #     lda_based_context.init_reviews()
        # else:
        #     text_reviews = []
        #     for record in self.records:
        #         text_reviews.append(record['text'])
        #     lda_based_context = LdaBasedContext(text_reviews)
        #     lda_based_context.init_reviews()
        self.context_rich_topics = lda_based_context.get_context_rich_topics()

        self.lda_model = lda_based_context.topic_model
        print('building similarity matrix', time.strftime("%H:%M:%S"))
        self.context_matrix = self.create_context_matrix(records)
        self.similarity_matrix = self.create_similarity_matrix()
        print('finished building similarity matrix', time.strftime("%H:%M:%S"))

    def get_rating(self, user, item):
        return self.ratings_matrix[user][item]

    def get_rating_on_context(self, user, item, context, threshold):

        neighbour_context = self.context_matrix[user][item]
        context_similarity = get_context_similarity(
            context, neighbour_context, self.context_rich_topics)

        if context_similarity < threshold:
            return None

        return self.get_rating(user, item)

    def create_similarity_matrix(self):

        similarity_matrix = {}

        for user in self.user_ids:
            similarity_matrix[user] = {}

        for user_id1, user_id2 in itertools.combinations(self.user_ids, 2):
            similarity = self.calculate_user_similarity(user_id1, user_id2, 0)
            similarity_matrix[user_id1][user_id2] = similarity
            similarity_matrix[user_id2][user_id1] = similarity

        return similarity_matrix

    def get_topic_distribution(self, review):
        """

        :type review: str
        """
        review_bow = lda_context_utils.create_bag_of_words([review])
        dictionary = corpora.Dictionary(review_bow)
        corpus = dictionary.doc2bow(review_bow[0])
        lda_corpus = self.lda_model.get_document_topics(corpus)

        topic_distribution =\
            lda_document_to_topic_distribution(lda_corpus, self.num_topics)

        return topic_distribution

    # TODO: Adapt this to a data structure in which a user can rate the same
    # item multiple times in different contexts
    def calculate_neighbour_contribution(
            self, neighbour_id, item_id, context, threshold):

        neighbour_rating = self.get_rating_on_context(
            neighbour_id, item_id, context, threshold)
        neighbor_average =\
            self.calculate_user_baseline(neighbour_id, context, threshold)

        return neighbour_rating - neighbor_average

    def calculate_user_baseline(self, user_id, context, threshold):
        user_rated_items = self.user_dictionary[user_id].item_ratings
        num_rated_items = len(user_rated_items)

        ratings_sum = 0.0
        num_ratings = 0.0
        for item_id in user_rated_items.keys():
            rating = self.get_rating_on_context(
                user_id, item_id, context, threshold)
            if rating:
                ratings_sum += rating
                num_ratings += 1

        user_baseline = ratings_sum / num_rated_items
        return user_baseline

    def calculate_user_similarity(self, user1, user2, threshold):

        common_items = self.get_common_rated_items(user1, user2)

        if not common_items:
            return None

        filtered_items = {}

        for item in common_items:
            context1 = self.context_matrix[user1][item]
            context2 = self.context_matrix[user2][item]
            similarity = get_context_similarity(
                context1, context2, self.context_rich_topics)
            if similarity > threshold:
                filtered_items[item] = similarity

        numerator = 0
        denominator1 = 0
        denominator2 = 0
        denominator3 = 0
        user1_average = self.user_dictionary[user1].average_overall_rating
        user2_average = self.user_dictionary[user2].average_overall_rating

        for item in filtered_items.keys():
            similarity = filtered_items[item]
            user1_rating = self.get_rating(user1, item)
            user2_rating = self.get_rating(user2, item)

            numerator +=\
                (user1_rating - user1_average) *\
                (user2_rating - user2_average) *\
                similarity
            denominator1 += (user1_rating - user1_average) ** 2
            denominator2 += (user2_rating - user2_average) ** 2
            denominator3 += similarity ** 2

        denominator = math.sqrt(denominator1 * denominator2 * denominator3)

        if denominator == 0:
            return 0

        return numerator / denominator

    def get_common_rated_items(self, user1, user2):
        """
        Obtains the items that user1 and user2 have rated in common

        :param user1:
        :param user2:
        """
        items_user1 = self.user_dictionary[user1].item_ratings.keys()
        items_user2 = self.user_dictionary[user2].item_ratings.keys()

        return list(set(items_user1).intersection(items_user2))

    def get_neighbourhood(self, user, item, context, threshold):

        all_users = self.user_ids[:]
        all_users.remove(user)
        neighbours = []

        # We remove the users who have not rated the given item
        for neighbour in all_users:
            if item in self.reviews_matrix[neighbour]:
                neighbours.append(neighbour)

        neighbour_similarity_map = {}
        for neighbour in neighbours:
            neighbour_context = self.context_matrix[neighbour][item]
            context_similarity = get_context_similarity(
                context, neighbour_context, self.context_rich_topics)
            if context_similarity > threshold:
                neighbour_similarity_map[neighbour] = context_similarity

        # Sort the users by similarity
        neighbourhood = dictionary_utils.sort_dictionary_keys(
            neighbour_similarity_map)  # [:self.num_neighbors]

        return neighbourhood

    def get_neighbourhood2(self, user, item, context, threshold):

        sim_users_matrix = self.similarity_matrix[user].copy()
        sim_users_matrix.pop(user, None)

        # We remove the users who have not rated the given item
        sim_users_matrix = {
            k: v for k, v in sim_users_matrix.items()
            if item in self.ratings_matrix[k]}

        # We remove neighbours that don't have a similarity with user
        sim_users_matrix = {
            k: v for k, v in sim_users_matrix.items()
            if v}

        # Sort the users by similarity
        neighbourhood = dictionary_utils.sort_dictionary_keys(
            sim_users_matrix)
        if self.num_neighbors:
            return neighbourhood[:self.num_neighbors]

        return neighbourhood

    def predict_rating(self, user, item, review):

        if user not in self.user_ids:
            return None

        threshold1 = 0.0
        threshold2 = 0.0
        threshold3 = 0.0
        threshold4 = 0.0

        ratings_sum = 0
        similarities_sum = 0
        num_users = 0
        user_context = self.get_topic_distribution(review)
        neighbourhood =\
            self.get_neighbourhood2(user, item, user_context, threshold1)

        if not neighbourhood:
            return None

        # print('neighbourhood size', len(neighbourhood))

        num_neighbours = 0

        for neighbour in neighbourhood:

            similarity = \
                self.calculate_user_similarity(user, neighbour, threshold4)

            if (item in self.user_dictionary[neighbour].item_ratings and
                        similarity is not None):

                num_neighbours += 1

                neighbour_contribution = self.calculate_neighbour_contribution(
                    neighbour, item, user_context, threshold2)
                ratings_sum += similarity * neighbour_contribution
                similarities_sum += abs(similarity)
                num_users += 1

            if num_users == self.num_neighbors:
                break

        # print('used neighbours', num_neighbours)

        if similarities_sum == 0:
            return None

        k = 1 / similarities_sum
        user_average = \
            self.calculate_user_baseline(user, user_context, threshold3)

        predicted_rating = user_average + k * ratings_sum

        return predicted_rating

    def create_context_matrix(self, records):
        context_matrix = {}

        for record in records:
            user = record['user_id']
            item = record['offering_id']
            review_text = record['text']
            topic_distribution = self.get_topic_distribution(review_text)

            if user not in context_matrix:
                context_matrix[user] = {}
            context_matrix[user][item] = topic_distribution

        return context_matrix


def load_data(json_file):
    records = ETLUtils.load_json_file(json_file)
    fields = ['user_id', 'business_id', 'stars', 'text', 'review_id']
    records = ETLUtils.select_fields(fields, records)

    # We rename the 'stars' field to 'overall_rating' to take advantage of the
    # function extractor.get_user_average_overall_rating
    for record in records:
        record['overall_rating'] = record.pop('stars')
        record['offering_id'] = record.pop('business_id')

    return records


def create_reviews_matrix(records):
    """
    Creates a dictionary of dictionaries with all the ratings available in the
    records. A rating can then be accessed by using ratings_matrix[user][item].
    The goal of this method is to generate a data structure in which the ratings
    can be queried in constant time.

    Beware that this method assumes that there is only one rating per user-item
    pair, in case there is more than one, only the last rating found in the
    records list will be present on the matrix, the rest will be ignored

    :type records: list[dict]
    :param records: a list of  dictionaries, in which each record contains three
    fields: 'user_id', 'business_id' and 'rating'
    :rtype: dict[dict]
    :return: a dictionary of dictionaries with all the ratings
    """
    reviews_matrix = {}

    for record in records:
        user = record['user_id']
        item = record['offering_id']
        rating = record['text']

        if user not in reviews_matrix:
            reviews_matrix[user] = {}
        reviews_matrix[user][item] = rating

    return reviews_matrix


def get_context_similarity(context1, context2, context_rich_topics):

    # We filter the topic model, selecting only the topics that contain context
    filtered_context1 =\
        numpy.array([context1[i[0]] for i in context_rich_topics])
    filtered_context2 =\
        numpy.array([context2[i[0]] for i in context_rich_topics])

    return 1 / (1 + numpy.linalg.norm(filtered_context1-filtered_context2))


def lda_document_to_topic_distribution(lda_document, num_topics):

    topic_distribution = numpy.zeros(num_topics)
    for pair in lda_document:
        topic_distribution[pair[0]] = pair[1]
    return topic_distribution




def main():
    # reviews_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review_hotels.json"
    # reviews_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_hotels_shuffled.json"
    reviews_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_restaurants_shuffled.json"
    # reviews_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_spas_shuffled.json"
    # my_records = context_utils.load_reviews(reviews_file)
    my_records = load_data(reviews_file)
    print("records:", len(my_records))
    my_num_topics = 150

    print("\n***************************\n")

    # my_records = load_data(reviews_file)
    # my_records = extractor.remove_users_with_low_reviews(my_records, 200)
    my_records = extractor.remove_users_with_low_reviews(my_records, 200)
    # shuffle(my_records)

    my_index = 0
    my_reviews = []
    for record in my_records:
        my_index += 1
        review = Review(record['text'])
        review.id = record['review_id']
        my_reviews.append(review)
        print('index', my_index)
    # my_file = '/Users/fpena/UCC/Thesis/datasets/context/reviews_context_restaurants_200.pkl'
    # my_file = '/Users/fpena/UCC/Thesis/datasets/context/reviews_context_hotel_2.pkl'
    # my_file = '/Users/fpena/tmp/reviews_hotel_shuffled.pkl'
    # my_file = '/Users/fpena/UCC/Thesis/datasets/context/reviews_spa_shuffled_2.pkl'
    # with open(my_file, 'wb') as write_file:
    #     pickle.dump(my_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    # with open(my_file, 'rb') as read_file:
    #     my_cache_reviews = pickle.load(read_file)

    # context_knn = ContextKnn(my_num_topics)

    # tknc = TopKNeighbourhoodCalculator()
    # nc = ContextNeighbourhoodCalculator()
    # ncc = NeighbourContributionCalculator()
    # ubc = UserBaselineCalculator()
    # usc = PBCSimilarityCalculator()
    # cosine_usc = CBCSimilarityCalculator()

    # contextual_knn2 = ContextualKNN(my_num_topics, tknc, ncc, ubc, usc, my_reviews)

    # bnc = BasicNeighbourhoodCalculator()
    # bncc = BasicNeighbourContributionCalculator()
    # bubc = BasicUserBaselineCalculator()
    # busc = BasicUserSimilarityCalculator()
    #
    # snc = SimpleNeighbourhoodCalculator()
    # chnc = ContextHybridNeighbourhoodCalculator()
    #
    # contextual_knn = ContextualKNN(my_num_topics, nc, ncc, ubc, usc, has_context=True)
    # contextual_knn2 = ContextualKNN(my_num_topics, nc, ncc, ubc, busc, has_context=True)
    # contextual_knn3 = ContextualKNN(my_num_topics, bnc, bncc, bubc, busc)
    # contextual_knn4 = ContextualKNN(my_num_topics, nc, ncc, ubc, cosine_usc, has_context=True)
    # basic_contextual_knn = BasicContextualKNN(my_num_topics, bnc, bncc, bubc, busc)

    # contextual_knn.threshold1 = 0.8
    # contextual_knn.threshold2 = 0.8
    # contextual_knn.threshold3 = 0.8
    # contextual_knn.threshold4 = 0.8

    # print('Context KNN')
    # context_knn.load(my_records)
    # recommender_evaluator.perform_cross_validation(my_records, context_knn, 5, True)
    # basic_knn_rec = BasicKNN(None)
    # print('Basic KNN')
    # recommender_evaluator.perform_cross_validation(my_records, basic_knn_rec, 5)
    # print('Context KNN')
    # recommender_evaluator.perform_cross_validation(my_records, context_knn, 5, my_cache_reviews)
    # print('Contextual KNN')
    # recommender_evaluator.perform_cross_validation(my_records, contextual_knn, 5, my_cache_reviews)
    # print('Contextual KNN2')
    # recommender_evaluator.perform_cross_validation(my_records, contextual_knn2, 5, my_cache_reviews)
    # print('Contextual KNN3')
    # recommender_evaluator.perform_cross_validation(my_records, contextual_knn3, 5)
    # recommender_evaluator.perform_cross_validation(my_records, contextual_knn3, 5)
    # recommender_evaluator.perform_cross_validation(my_records, contextual_knn3, 5, True, my_cache_reviews)
    # precision_in_top_n.calculate_recall_in_top_n(my_records, basic_contextual_knn, 10, 65)
    # precision_in_top_n.calculate_recall_in_top_n(my_records, basic_knn_rec, 10, 65)
    # print('Basic KNN')
    # precision_in_top_n.calculate_recall_in_top_n(my_records, basic_knn_rec, 10, 5, 5.0, my_cache_reviews)
    # precision_in_top_n.calculate_recall_in_top_n(my_records, basic_knn_rec, 10, 5, 5.0, False, my_cache_reviews)
    # print('Context KNN')
    # precision_in_top_n.calculate_recall_in_top_n(my_records, context_knn, 10, 5, 5.0, my_cache_reviews)
    # print('Contextual KNN')
    # precision_in_top_n.calculate_recall_in_top_n(my_records, contextual_knn, 10, 5, 5.0, my_cache_reviews)
    # print('Contextual KNN 2')
    # precision_in_top_n.calculate_recall_in_top_n(my_records, contextual_knn2, 10, 5, 5.0, my_cache_reviews)
    # print('Contextual KNN 3')
    # precision_in_top_n.calculate_recall_in_top_n(my_records, contextual_knn3, 10, 5, 5.0, my_cache_reviews)
    # print('Contextual KNN 4')
    # precision_in_top_n.calculate_recall_in_top_n(my_records, contextual_knn4, 10, 5, 5.0, my_cache_reviews)
    # print('Basic Contextual KNN')
    # precision_in_top_n.calculate_recall_in_top_n(my_records, basic_contextual_knn, 10, 5, 5.0, True, my_cache_reviews)
    # precision_in_top_n.calculate_recall_in_top_n(my_records, context_knn, 10, 65, 5.0, True)
    # precision_in_top_n.calculate_recall_in_top_n(my_records, contextual_knn2, 10, 65, 5.0, True)

    # lda_based_context = LdaBasedContext(my_reviews)
    # lda_based_context.init_reviews()
    # my_topics = lda_based_context.get_context_rich_topics()
    # print(my_topics)

numpy.random.seed(0)
start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
