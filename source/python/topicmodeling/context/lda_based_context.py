import time
import operator
from gensim import corpora
from topicmodeling.context import lda_context_utils
from topicmodeling.context import context_utils
from topicmodeling.context.review import Review


__author__ = 'fpena'

class LdaBasedContext:

    def __init__(self, text_reviews):
        self.text_reviews = text_reviews
        self.alpha = 0.005
        self.beta = 1.0
        self.reviews = None
        self.specific_reviews = None
        self.generic_reviews = None
        self.all_nouns = None
        self.all_senses = None
        self.sense_groups = None
        self.review_topics_list = None
        self.num_topics = 50
        self.topics = range(self.num_topics)

    def init_reviews(self):

        print('init_reviews', time.strftime("%H:%M:%S"))

        self.reviews = []
        self.specific_reviews = []
        self.generic_reviews = []

        for text_review in self.text_reviews:
            self.reviews.append(Review(text_review))

        text_specific_reviews, text_generic_reviews =\
            context_utils.cluster_reviews(self.text_reviews)

        for text_review in text_specific_reviews:
            self.specific_reviews.append(Review(text_review))
        for text_review in text_generic_reviews:
            self.generic_reviews.append(Review(text_review))

        # self.all_nouns = context_utils.get_all_nouns(self.reviews)

    def filter_topics(self):

        specific_reviews_text =\
            context_utils.get_text_from_reviews(self.specific_reviews)
        generic_reviews_text =\
            context_utils.get_text_from_reviews(self.generic_reviews)
        topic_model = lda_context_utils.discover_topics(specific_reviews_text, self.num_topics)

        specific_bag_of_words =\
            lda_context_utils.create_bag_of_words(specific_reviews_text)
        generic_bag_of_words =\
            lda_context_utils.create_bag_of_words(generic_reviews_text)

        dictionary = corpora.Dictionary(specific_bag_of_words)
        dictionary.filter_extremes(2, 0.6)
        specific_corpus = [dictionary.doc2bow(text) for text in specific_bag_of_words]

        dictionary = corpora.Dictionary(generic_bag_of_words)
        dictionary.filter_extremes(2, 0.6)
        generic_corpus = [dictionary.doc2bow(text) for text in generic_bag_of_words]

        specific_topics = topic_model[specific_corpus]
        generic_topics = topic_model[generic_corpus]

        lda_context_utils.update_reviews_with_topics(
            specific_topics, self.specific_reviews)
        lda_context_utils.update_reviews_with_topics(
            generic_topics, self.generic_reviews)

        topic_ratio_map = {}

        for topic in range(self.num_topics):
            specific_weighted_frq = \
                lda_context_utils.calculate_topic_weighted_frequency(
                    topic, self.specific_reviews)
            generic_weighted_frq = \
                lda_context_utils.calculate_topic_weighted_frequency(
                    topic, self.generic_reviews)

            if generic_weighted_frq < self.alpha or specific_weighted_frq < self.alpha:
                self.topics.remove(topic)
                continue

            ratio = specific_weighted_frq / generic_weighted_frq

            # if ratio < self.beta:
            #     self.topics.remove(topic)
            #     continue

            topic_ratio_map[topic] = ratio

        sorted_topics = sorted(
            topic_ratio_map.items(), key=operator.itemgetter(1), reverse=True)

        # for topic in topic_model.show_topics(num_topics=self.num_topics):
        #     print(topic)
        for i in range(topic_model.num_topics):
            print('topic', i, topic_model.print_topic(i, topn=50))

        return sorted_topics











def main():
    reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review.json"
    my_reviews = context_utils.load_reviews(reviews_file)[:5000]
    print("reviews:", len(my_reviews))

    # lda_context_utils.discover_topics(my_reviews, 50)
    lda_based_context = LdaBasedContext(my_reviews)
    lda_based_context.init_reviews()
    my_topics = lda_based_context.filter_topics()
    print(my_topics)

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
