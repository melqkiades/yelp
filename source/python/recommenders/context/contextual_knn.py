import time
from topicmodeling.context import lda_context_utils
from topicmodeling.context.lda_based_context import LdaBasedContext
from tripadvisor.fourcity import extractor

# __author__ = 'fpena'


class ContextualKNN:

    def __init__(self, num_topics, neighbourhood_calculator,
                 neighbour_contribution_calculator, user_baseline_calculator,
                 user_similarity_calculator, reviews=None, has_context=False):
        self.user_ids = None
        self.user_dictionary = None
        self.num_topics = num_topics
        self.lda_model = None
        self.reviews = reviews
        self.num_neighbours = None
        self.context_rich_topics = None
        self.has_context = has_context

        self.neighbourhood_calculator = neighbourhood_calculator
        self.neighbour_contribution_calculator =\
            neighbour_contribution_calculator
        self.user_baseline_calculator = user_baseline_calculator
        self.user_similarity_calculator = user_similarity_calculator

        self.threshold1 = 0.0
        self.threshold2 = 0.0
        self.threshold3 = 0.0
        self.threshold4 = 0.0

    def load(self, records):
        # self.records = records
        self.user_dictionary = extractor.initialize_users(records, False)
        self.user_ids = extractor.get_groupby_list(records, 'user_id')

        if self.has_context:
            self.load_context(records)

        print('building similarity matrix', time.strftime("%H:%M:%S"))
        self.user_similarity_calculator.load(
            self.user_ids, self.user_dictionary, self.context_rich_topics)
        user_similarity_matrix = self.user_similarity_calculator.\
            create_similarity_matrix(self.threshold4)
        print('finished building similarity matrix', time.strftime("%H:%M:%S"))

        self.neighbourhood_calculator.load(
            self.user_ids, self.user_dictionary, self.context_rich_topics,
            self.num_neighbours, user_similarity_matrix)
        self.user_baseline_calculator.load(
            self.user_dictionary, self.context_rich_topics)
        self.neighbour_contribution_calculator.load(
            self.user_baseline_calculator)

    def load_context(self, records):
        if self.reviews:
            lda_based_context = LdaBasedContext()
            lda_based_context.reviews = self.reviews
            lda_based_context.init_reviews()
        else:
            text_reviews = []
            for record in records:
                text_reviews.append(record['text'])
            lda_based_context = LdaBasedContext(text_reviews)
            lda_based_context.init_reviews()
        self.context_rich_topics = lda_based_context.get_context_rich_topics()

        self.lda_model = lda_based_context.topic_model

        for user in self.user_dictionary.values():
            user.item_contexts = lda_context_utils.get_user_item_contexts(
                records, self.lda_model, user.user_id, True
            )


    # def get_topic_distribution(self, review):
    #     """
    #
    #     :type review: str
    #     """
    #     review_bow = lda_context_utils.create_bag_of_words([review])
    #     dictionary = corpora.Dictionary(review_bow)
    #     corpus = dictionary.doc2bow(review_bow[0])
    #     lda_corpus = self.lda_model.get_document_topics(corpus)
    #
    #     topic_distribution = self.lda_document_to_topic_distribution(lda_corpus)
    #
    #     return topic_distribution

    def predict_rating(self, user, item, review=None):

        # print('predict_rating', user, item)

        if user not in self.user_ids:
            return None

        ratings_sum = 0
        similarities_sum = 0
        num_users = 0
        user_context = None
        if self.has_context:
            user_context =\
                lda_context_utils.get_topic_distribution(review, self.lda_model)
        neighbourhood = self.neighbourhood_calculator.get_neighbourhood(
            user, item, user_context, self.threshold1)

        if not neighbourhood:
            return None

        num_neighbours = 0

        for neighbour in neighbourhood:

            similarity =\
                self.user_similarity_calculator.calculate_user_similarity(
                    user, neighbour, self.threshold4)

            if (item in self.user_dictionary[neighbour].item_ratings and
                    similarity is not None):

                if similarity <= 0:
                    continue

                num_neighbours += 1

                neighbour_contribution = self.neighbour_contribution_calculator.\
                    calculate_neighbour_contribution(
                        neighbour, item, user_context, self.threshold2)

                if neighbour_contribution is None:
                    continue

                ratings_sum += similarity * neighbour_contribution
                similarities_sum += abs(similarity)
                num_users += 1

        if similarities_sum == 0:
            return None

        k = 1 / similarities_sum
        user_average = self.user_baseline_calculator.calculate_user_baseline(
            user, user_context, self.threshold3)

        predicted_rating = user_average + k * ratings_sum

        # print('num used neighbours', num_users)

        return predicted_rating

    # def lda_document_to_topic_distribution(self, lda_document):
    #
    #     topic_distribution = numpy.zeros(self.num_topics)
    #     for pair in lda_document:
    #         topic_distribution[pair[0]] = pair[1]
    #     return topic_distribution

    # def create_context_matrix(self, records):
    #     context_matrix = {}
    #
    #     for record in records:
    #         user = record['user_id']
    #         item = record['offering_id']
    #         review_text = record['text']
    #         topic_distribution = lda_context_utils.get_topic_distribution(review_text, self.lda_model)
    #
    #         if user not in context_matrix:
    #             context_matrix[user] = {}
    #         context_matrix[user][item] = topic_distribution
    #
    #     return context_matrix


# def create_reviews_matrix(records):
#     """
#     Creates a dictionary of dictionaries with all the ratings available in the
#     records. A rating can then be accessed by using ratings_matrix[user][item].
#     The goal of this method is to generate a data structure in which the ratings
#     can be queried in constant time.
#
#     Beware that this method assumes that there is only one rating per user-item
#     pair, in case there is more than one, only the last rating found in the
#     records list will be present on the matrix, the rest will be ignored
#
#     :type records: list[dict]
#     :param records: a list of  dictionaries, in which each record contains three
#     fields: 'user_id', 'business_id' and 'rating'
#     :rtype: dict[dict]
#     :return: a dictionary of dictionaries with all the ratings
#     """
#     reviews_matrix = {}
#
#     for record in records:
#         user = record['user_id']
#         item = record['offering_id']
#         rating = record['text']
#
#         if user not in reviews_matrix:
#             reviews_matrix[user] = {}
#         reviews_matrix[user][item] = rating
#
#     return reviews_matrix


# def filter_context_topics_by_ratio(context_topics):
#
#     filtered_topics = []
#     for topic in context_topics:
#         ratio = topic[1]
#         if ratio > 1.0:
#             filtered_topics.append(topic)
#
#     return filtered_topics
