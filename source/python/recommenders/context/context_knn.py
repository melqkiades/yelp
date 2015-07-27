from gensim import corpora
import itertools
import math
import numpy
import time
import scipy
from etl import ETLUtils
from recommenders.context import basic_knn
from recommenders.context.basic_knn import BasicKNN
from topicmodeling.context import lda_context_utils
from topicmodeling.context import context_utils
from scipy.special import gammaln, psi
from topicmodeling.context.lda_based_context import LdaBasedContext
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity.user import User
from utils import dictionary_utils
from tripadvisor.fourcity import recommender_evaluator

__author__ = 'fpena'


class ContextKnn:

    def __init__(self, num_topics):
        self.reviews = None
        self.num_neighbors = None
        self.ratings_matrix = None
        self.reviews_matrix = None
        self.similarity_matrix = None
        self.user_dictionary = None
        self.user_ids = None
        self.num_topics = num_topics
        self.lda_model = None
        self.dictionary = None
        self.context_topics = None
        self.topic_indices = None

    def load(self, reviews):
        self.reviews = reviews
        self.ratings_matrix = basic_knn.create_ratings_matrix(reviews)
        self.reviews_matrix = create_reviews_matrix(reviews)
        self.user_dictionary = extractor.initialize_users(self.reviews, False)
        self.user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        text_reviews = []
        for record in self.reviews:
            text_reviews.append(record['text'])
        # self.lda_model =\
        #     lda_context_utils.discover_topics(text_reviews, self.num_topics)
        lda_based_context = LdaBasedContext(text_reviews)
        lda_based_context.init_reviews()
        self.context_topics = lda_based_context.filter_topics()
        self.topic_indices =\
            self.filter_context_topics_by_ratio(self.context_topics)

        self.lda_model = lda_based_context.topic_model
        print('building similarity matrix')
        self.similarity_matrix = self.create_similarity_matrix()

    def filter_context_topics_by_ratio(self, context_topics):

        filtered_topics = []
        for topic in context_topics:
            ratio = topic[1]
            if ratio > 1.0:
                filtered_topics.append(topic)

        return filtered_topics

    def get_rating(self, user, item):
        return self.ratings_matrix[user][item]

    def create_similarity_matrix(self):

        similarity_matrix = {}

        for user in self.user_ids:
            similarity_matrix[user] = {}

        for user_id1, user_id2 in itertools.combinations(self.user_ids, 2):
            similarity = self.calculate_user_similarity(user_id1, user_id2, 0)
            similarity_matrix[user_id1][user_id2] = similarity
            similarity_matrix[user_id2][user_id1] = similarity
            # print('similarity', similarity)

        return similarity_matrix


    def train(self, reviews):
        """

        :type reviews: list[str]
        :param reviews:
        """
        self.lda_model =\
            lda_context_utils.discover_topics(reviews, self.num_topics)

    def get_topic_distribution(self, review):
        """

        :type review: str
        """
        review_bow = lda_context_utils.create_bag_of_words([review])
        dictionary = corpora.Dictionary(review_bow)
        corpus = dictionary.doc2bow(review_bow[0])
        lda_corpus = self.lda_model[corpus]

        topic_distribution =\
            lda_document_to_topic_distribution(lda_corpus, self.num_topics)

        return topic_distribution

    def get_topic_distribution2(self, review):
        """

        :type review: str
        """
        review_bow = lda_context_utils.create_bag_of_words([review])
        dictionary = corpora.Dictionary(review_bow)
        corpus = dictionary.doc2bow(review_bow[0])
        lda_corpus = infer(corpus, self.lda_model)

        topic_distribution =\
            lda_document_to_topic_distribution(lda_corpus, 5)

        return topic_distribution

    # TODO: Adapt this to a data structure in which a user can rate the same
    # item multiple times in different contexts
    def calculate_neighbour_contribution(self, user_id, item_id, threshold):
        return self.get_rating(user_id, item_id)

    def calculate_user_baseline(self, user_id, threshold):
        user_rated_items = self.user_dictionary[user_id].item_ratings
        num_rated_items = len(user_rated_items)

        ratings_sum = 0.0
        for item_id in user_rated_items.keys():
            ratings_sum += self.calculate_neighbour_contribution(
                user_id, item_id, threshold)

        user_baseline = ratings_sum / num_rated_items
        return user_baseline

    def calculate_user_similarity(self, user1, user2, threshold):

        common_items = self.get_common_rated_items(user1, user2)

        if not common_items:
            return None

        filtered_items = []

        for item in common_items:
            review1 = self.reviews_matrix[user1][item]
            review2 = self.reviews_matrix[user2][item]
            similarity = self.calculate_review_similarity(review1, review2)
            if similarity > threshold:
                filtered_items.append(item)

        numerator = 0
        denominator1 = 0
        denominator2 = 0
        denominator3 = 0
        user1_average = self.user_dictionary[user1].average_overall_rating
        user2_average = self.user_dictionary[user2].average_overall_rating

        for item in filtered_items:
            review1 = self.reviews_matrix[user1][item]
            review2 = self.reviews_matrix[user2][item]
            similarity = self.calculate_review_similarity(review1, review2)
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

        # numerator = 0
        # denominator1 = 0
        # denominator2 = 0
        #
        # for item in common_items:
        #     user1_rating = self.get_rating(user1, item)
        #     user2_rating = self.get_rating(user2, item)
        #
        #     numerator += user1_rating * user2_rating
        #     denominator1 += user1_rating ** 2
        #     denominator2 += user2_rating ** 2
        #
        # denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

        # if denominator == 0:
        #     pass

        return numerator / denominator

        # similarity = numerator / denominator
        # return similarity

    def get_common_rated_items(self, user1, user2):
        """
        Obtains the items that user1 and user2 have rated in common

        :param user1:
        :param user2:
        """
        items_user1 = self.user_dictionary[user1].item_ratings.keys()
        items_user2 = self.user_dictionary[user2].item_ratings.keys()

        return list(set(items_user1).intersection(items_user2))

    def calculate_review_similarity(self, review1, review2):

        topic_dist1 = self.get_topic_distribution(review1)
        topic_dist2 = self.get_topic_distribution(review2)

        return get_document_similarity(topic_dist1, topic_dist2, self.topic_indices)

    def calculate_review_similarity2(self, review1, review2):
        topic_dist1 = self.get_topic_distribution2(review1)
        topic_dist2 = self.get_topic_distribution2(review2)

        # print(topic_dist1)

        return get_document_similarity(topic_dist1, topic_dist2)

    def get_neighbourhood(self, user, item, review, threshold):

        neighbours2 = self.user_ids[:]
        neighbours2.remove(user)
        neighbours = []

        # print('item', item)

        # We remove the users who have not rated the given item
        for neighbour in neighbours2:
            # print(self.reviews_matrix[neighbour].keys())
            if item in self.reviews_matrix[neighbour]:
                neighbours.append(neighbour)

        neighbour_similarity_map = {}
        for neighbour in neighbours:
            neighbour_review = self.reviews_matrix[neighbour][item]
            context_similarity =\
                self.calculate_review_similarity(review, neighbour_review)
            if context_similarity > threshold:
                neighbour_similarity_map[neighbour] = context_similarity

        # print(neighbour_similarity_map)

        # Sort the users by similarity
        neighbourhood = dictionary_utils.sort_dictionary_keys(
            neighbour_similarity_map)  # [:self.num_neighbors]

        # print('neighbourhood size:', len(neighbourhood))

        return neighbourhood

    def predict_rating(self, user, item, review):

        if user not in self.user_ids:
            return None

        threshold1 = 0
        threshold2 = 0
        threshold3 = 0
        threshold4 = 0

        ratings_sum = 0
        similarities_sum = 0
        num_users = 0
        neighbourhood = self.get_neighbourhood(user, item, review, threshold1)

        if not neighbourhood:
            return None

        # print(neighbourhood)

        for neighbour in neighbourhood:

            similarity =\
                self.calculate_user_similarity(user, neighbour, threshold4)

            if item in self.user_dictionary[neighbour].item_ratings and similarity is not None:

                neighbor_rating = self.calculate_neighbour_contribution(
                    neighbour, item, threshold2)
                neighbor_average = \
                    self.calculate_user_baseline(neighbour, threshold2)
                ratings_sum += similarity * (neighbor_rating - neighbor_average)
                similarities_sum += abs(similarity)
                num_users += 1

                print('similarity', similarity)

            if num_users == self.num_neighbors:
                break

        if similarities_sum == 0:
            return None

        k = 1 / similarities_sum
        user_average = self.calculate_user_baseline(user, threshold3)

        predicted_rating = user_average + k * ratings_sum

        return predicted_rating




def load_data(json_file):
    records = ETLUtils.load_json_file(json_file)
    fields = ['user_id', 'business_id', 'stars', 'text']
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


def get_document_similarity(document1, document2, topic_indices):

    filtered_document1 = numpy.array([document1[i[0]] for i in topic_indices])
    filtered_document2 = numpy.array([document2[i[0]] for i in topic_indices])

    return 1 / (1 + numpy.linalg.norm(filtered_document1-filtered_document2))


def lda_document_to_topic_distribution(lda_document, num_topics):

    topic_distribution = numpy.zeros((num_topics))
    for pair in lda_document:
        topic_distribution[pair[0]] = pair[1]
    return topic_distribution


def infer(bow, lda_model, eps=0.01):


        num_topics = lda_model.num_topics
        expElogbeta = lda_model.expElogbeta
        alpha = lda_model.alpha
        gamma_threshold = lda_model.gamma_threshold
        iterations = lda_model.iterations

        num_topics = 5
        expElogbeta = numpy.delete(expElogbeta, [1, 3, 5, 7, 9], axis=0)
        alpha = numpy.delete(alpha, [1, 3, 5, 7, 9])

        gamma, _ = inference(
            [bow], num_topics, expElogbeta, alpha, gamma_threshold, iterations)
        topic_dist = gamma[0] / sum(gamma[0]) # normalize to proper distribution
        return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
                if topicvalue >= eps] # ignore document's topics that have prob < eps


def inference(chunk, num_topics, expElogbeta, alpha, gamma_threshold, iterations, collect_sstats=False):
        """
        Given a chunk of sparse document vectors, estimate gamma (parameters
        controlling the topic weights) for each document in the chunk.

        This function does not modify the model (=is read-only aka const). The
        whole input chunk of document is assumed to fit in RAM; chunking of a
        large corpus must be done earlier in the pipeline.

        If `collect_sstats` is True, also collect sufficient statistics needed
        to update the model's topic-word distributions, and return a 2-tuple
        `(gamma, sstats)`. Otherwise, return `(gamma, None)`. `gamma` is of shape
        `len(chunk) x self.num_topics`.

        Avoids computing the `phi` variational parameter directly using the
        optimization presented in **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.

        """
        try:
            _ = len(chunk)
        except:
            chunk = list(chunk) # convert iterators/generators to plain list, so we have len() etc.
        if len(chunk) > 1:
            print("performing inference on a chunk of %i documents" % len(chunk))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = numpy.random.gamma(100., 1. / 100., (len(chunk), num_topics))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = numpy.exp(Elogtheta)
        if collect_sstats:
            sstats = numpy.zeros_like(expElogbeta)
        else:
            sstats = None
        converged = 0

        # Now, for each document d update that document's gamma and phi
        # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
        # Lee&Seung trick which speeds things up by an order of magnitude, compared
        # to Blei's original LDA-C code, cool!).
        for d, doc in enumerate(chunk):
            ids = [id for id, _ in doc]
            cts = numpy.array([cnt for _, cnt in doc])
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
            # phinorm is the normalizer.
            phinorm = numpy.dot(expElogthetad, expElogbetad) + 1e-100 # TODO treat zeros explicitly, instead of adding eps?

            # Iterate between gamma and phi until convergence
            for _ in xrange(iterations):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = alpha + expElogthetad * numpy.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = numpy.exp(Elogthetad)
                phinorm = numpy.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = numpy.mean(abs(gammad - lastgamma))
                if (meanchange < gamma_threshold):
                    converged += 1
                    break
            gamma[d, :] = gammad
            if collect_sstats:
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                sstats[:, ids] += numpy.outer(expElogthetad.T, cts / phinorm)

        if len(chunk) > 1:
            print("%i/%i documents converged within %i iterations" %
                (converged, len(chunk), iterations))

        if collect_sstats:
            # This step finishes computing the sufficient statistics for the
            # M step, so that
            # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
            # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
            sstats *= expElogbeta
        return gamma, sstats

def dirichlet_expectation(alpha):
    """
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.

    """
    if (len(alpha.shape) == 1):
        result = psi(alpha) - psi(numpy.sum(alpha))
    else:
        result = psi(alpha) - psi(numpy.sum(alpha, 1))[:, numpy.newaxis]
    return result.astype(alpha.dtype) # keep the same precision as input


def initialize_users(reviews, is_multi_criteria):
    """
    Builds a dictionary containing all the users in the reviews. Each user
    contains information about its average overall rating, the list of reviews
    that user has made, and the cluster the user belongs to

    :param reviews: the list of reviews
    :return: a dictionary with the users initialized, the keys of the
    dictionaries are the users' ID
    """
    user_ids = extractor.get_groupby_list(reviews, 'user_id')
    user_dictionary = {}

    for user_id in user_ids:
        user = User(user_id)
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
        user.average_overall_rating = extractor.get_user_average_overall_rating(
            user_reviews, user_id, apply_filter=False)
        user.item_ratings = get_user_item_ratings(user_reviews, user_id)
        user_dictionary[user_id] = user

    return user_dictionary


def get_user_item_ratings(reviews, user_id, apply_filter=False):
    """
    Returns a dictionary that contains the items that the given user has rated,
    where the key of the dictionary is the ID of the item and the value is the
    rating that user_id has given to that item

    :param reviews: a list of reviews
    :param user_id: the ID of the user
    :param apply_filter: a boolean that indicates if the reviews have to be
    filtered by user_id or not. In other word this boolean indicates if the list
    contains reviews from several users or not. If it does contains reviews from
    other users, those have to be removed
    :return: a dictionary with the items that the given user has rated
    """

    if apply_filter:
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    else:
        user_reviews = reviews

    if not user_reviews:
        return {}

    items_ratings = {}

    for review in reviews:
        item_id = review['offering_id']
        text_review = review['text']
        rating = review['overall_rating']
        items_ratings[item_id] = (rating, text_review)

    return items_ratings


def calculate_pearson_similarity(list1, list2):

    if len(list1) != len(list2):
        raise ValueError('list1 and list2 should have the same length')

    if not list1:
        return 0

    numerator = 0
    denominator1 = 0
    denominator2 = 0

    list1_average = sum(list1) / float(len(list1))
    list2_average = sum(list2) / float(len(list2))

    for index in range(len(list1)):
        value1 = list1[index]
        value2 = list2[index]

        # print('user average', user1_average)

        numerator +=\
            (value1 - list1_average) * (value2 - list2_average)
        denominator1 += (value1 - list1_average) ** 2
        denominator2 += (value2 - list2_average) ** 2

    denominator = math.sqrt(denominator1 * denominator2)

    if denominator == 0:
        return 0

    return numerator / denominator






def main():
    reviews_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review_hotels.json"
    my_reviews = context_utils.load_reviews(reviews_file)
    print("reviews:", len(my_reviews))
    my_num_topics = 150

    # processed = lda_context_utils.create_bag_of_words(my_reviews)
    # dictionary2 = corpora.Dictionary(processed)
    #
    # print(my_reviews[0])
    # print(my_reviews[1])
    # review_bow1 = lda_context_utils.create_bag_of_words([my_reviews[0]])[0]
    # review_bow2 = lda_context_utils.create_bag_of_words([my_reviews[1]])[0]
    # dictionary = corpora.Dictionary([review_bow1, review_bow2])
    # # dictionary.filter_extremes(2, 0.6)
    # print('review bow', review_bow1)
    # my_dict1 = dictionary2.doc2bow(review_bow1)
    # my_dict2 = dictionary2.doc2bow(review_bow2)
    # my_dict3 = dictionary2.doc2bow(review_bow1)
    # # corpus = [dictionary.doc2bow(text) for text in processed]
    #
    # lda_model = lda_context_utils.discover_topics(my_reviews, my_num_topics)
    # # my_dict2 = l(review_bow1)
    #
    # review_lda1 = lda_model[my_dict1]
    # review_lda2 = lda_model[my_dict2]
    # review_lda3 = lda_model[my_dict3]
    # print(review_bow1)
    # print(my_dict1)
    # print(my_dict3)
    # print(review_lda1)
    # print(review_lda3)
    # print(type(review_lda1))
    # print(type(review_lda1[0]))
    # print(type(review_lda1[0][0]))
    # topic_d1 = lda_document_to_topic_distribution(review_lda1, my_num_topics)
    # topic_d2 = lda_document_to_topic_distribution(review_lda2, my_num_topics)
    # topic_d3 = lda_document_to_topic_distribution(review_lda3, my_num_topics)
    # # print(topic_d1)
    # # print(topic_d2)
    # print(get_document_similarity(topic_d1, topic_d2))
    # print(get_document_similarity(topic_d1, topic_d3))
    # print(get_document_similarity(topic_d1, topic_d1))

    print("\n***************************\n")

    my_records = load_data(reviews_file)
    my_records = extractor.remove_users_with_low_reviews(my_records, 5)

    context_knn = ContextKnn(my_num_topics)
    # context_knn.load(my_records)
    recommender_evaluator.perform_cross_validation(my_records, context_knn, 5, True)
    basic_knn_rec = BasicKNN(5)
    recommender_evaluator.perform_cross_validation(my_records, basic_knn_rec, 5)
    # context_knn.train(my_reviews)
    # print(context_knn.calculate_review_similarity(my_reviews[0], my_reviews[1]))
    # print(context_knn.calculate_review_similarity(my_reviews[1], my_reviews[1]))
    # print(context_knn.calculate_review_similarity2(my_reviews[0], my_reviews[1]))
    # print(context_knn.calculate_review_similarity2(my_reviews[1], my_reviews[1]))


    # print(get_document_similarity(topic_d1, topic_d2))

    # lda_based_context = LdaBasedContext(my_reviews)
    # lda_based_context.init_reviews()
    # my_topics = lda_based_context.filter_topics()
    # print(my_topics)

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)


# my_doc1 = numpy.array([1, 3, 5])
# my_doc2 = numpy.array([1, 3, 5])
#
# print(get_document_similarity(my_doc1, my_doc2))


# my_list1 = [1, 1, 1, 1, 2]
# my_list2 = [2, 4, 0, 8, -10]
#
# print(calculate_pearson_similarity(my_list1, my_list2))
# print(scipy.stats.pearsonr(my_list1, my_list2))
