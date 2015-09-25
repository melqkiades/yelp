from topicmodeling.context import lda_context_utils
from topicmodeling.context.lda_based_context import LdaBasedContext
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


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

        # print('building similarity matrix', time.strftime("%H:%M:%S"))
        self.user_similarity_calculator.load(
            self.user_ids, self.user_dictionary, self.context_rich_topics)
        # user_similarity_matrix = self.user_similarity_calculator.\
        #     create_similarity_matrix(self.threshold4)
        # print('finished building similarity matrix', time.strftime("%H:%M:%S"))

        self.neighbourhood_calculator.load(
            self.user_ids, self.user_dictionary, self.context_rich_topics,
            self.num_neighbours)
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

    def predict_rating(self, user, item, review=None):

        # print('predict_rating', user, item)

        if user not in self.user_ids:
            return None

        ratings_sum = 0
        similarities_sum = 0
        num_neighbours = 0
        user_context = None
        if self.has_context:
            user_context =\
                lda_context_utils.get_topic_distribution(review, self.lda_model)
        neighbourhood = self.neighbourhood_calculator.get_neighbourhood(
            user, item, user_context, self.threshold1)

        if not neighbourhood:
            return None

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
                num_neighbours += 1

                if num_neighbours == self.num_neighbours:
                    break

        # print('num neighbours', num_neighbours)

        if similarities_sum == 0:
            return None

        k = 1 / similarities_sum
        user_average = self.user_baseline_calculator.calculate_user_baseline(
            user, user_context, self.threshold3)

        if user_average is None:
            return None

        predicted_rating = user_average + k * ratings_sum

        # print('num used neighbours', num_users)

        return predicted_rating

    def clear(self):
        self.user_ids = None
        self.user_dictionary = None
        self.lda_model = None
        self.num_neighbours = None
        self.context_rich_topics = None

        self.neighbourhood_calculator.clear()
        self.neighbour_contribution_calculator.clear()
        self.user_baseline_calculator.clear()
        self.user_similarity_calculator.clear()
