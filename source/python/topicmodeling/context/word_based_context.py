import time
import cPickle as pickle
from topicmodeling.context import context_utils
from topicmodeling.context import reviews_clusterer

__author__ = 'fpena'


class WordBasedContext:

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

    def init_reviews(self):

        print('init_reviews', time.strftime("%H:%M:%S"))

        self.reviews = []
        self.specific_reviews = []
        self.generic_reviews = []

        # for text_review in self.text_reviews:
        #     self.reviews.append(Review(text_review))

        my_file = '/Users/fpena/tmp/reviews_hotel.pkl'
        # my_file = '/Users/fpena/tmp/reviews_restaurant.pkl'
        # my_file = '/Users/fpena/tmp/sentences_hotel.pkl'
        # with open(my_file, 'wb') as write_file:
        #     pickle.dump(self.reviews, write_file, pickle.HIGHEST_PROTOCOL)

        with open(my_file, 'rb') as read_file:
            self.reviews = pickle.load(read_file)

        # self.reviews = self.reviews
        # for review in self.reviews:
        #     print(review)

        cluster_labels = reviews_clusterer.cluster_reviews(self.reviews)
        review_clusters =\
            reviews_clusterer.split_list_by_labels(self.reviews, cluster_labels)
        # print(cluster_labels)

        self.specific_reviews = review_clusters[0]
        self.generic_reviews = review_clusters[1]

        self.all_nouns = context_utils.get_all_nouns(self.reviews)

        context_utils.generate_stats(self.specific_reviews, self.generic_reviews)

    def filter_nouns(self):
        print('filter_nouns', time.strftime("%H:%M:%S"))
        unwanted_nouns = set()

        for noun in list(self.all_nouns):
            specific_weighted_frq =\
                context_utils.calculate_word_weighted_frequency(
                    noun, self.specific_reviews)
            generic_weighted_frq =\
                context_utils.calculate_word_weighted_frequency(
                    noun, self.generic_reviews)

            # print('specific_weighted_frq', specific_weighted_frq)
            # print('generic_weighted_frq', generic_weighted_frq)

            if generic_weighted_frq < self.alpha or specific_weighted_frq < self.alpha:
                self.all_nouns.remove(noun)
                unwanted_nouns.add(noun)
                continue

            ratio = specific_weighted_frq / generic_weighted_frq

            if ratio < self.beta:
                self.all_nouns.remove(noun)
                unwanted_nouns.add(noun)
                continue

        print('remove_nouns', time.strftime("%H:%M:%S"))
        context_utils.remove_nouns_from_reviews(self.reviews, unwanted_nouns)

        print('generating_all_senses', time.strftime("%H:%M:%S"))
        for review in self.reviews:
            context_utils.generate_senses(review)
        print('generating_specific_senses', time.strftime("%H:%M:%S"))
        for review in self.specific_reviews:
            context_utils.generate_senses(review)
        print('generating_generic_senses', time.strftime("%H:%M:%S"))
        for review in self.generic_reviews:
            context_utils.generate_senses(review)

    def calculate_sense_group_ratios(self):

        self.init_reviews()
        self.filter_nouns()

        print('building_groups', time.strftime("%H:%M:%S"))
        self.sense_groups = context_utils.build_groups(self.all_nouns)
        print('calculating_weights', time.strftime("%H:%M:%S"))

        for sense_group in self.sense_groups:
            specific_weighted_frq =\
                context_utils.calculate_group_weighted_frequency(
                    sense_group, self.specific_reviews)
            generic_weighted_frq =\
                context_utils.calculate_group_weighted_frequency(
                    sense_group, self.generic_reviews)

            ratio = specific_weighted_frq / generic_weighted_frq
            sense_group.ratio = ratio

        print('sorting', time.strftime("%H:%M:%S"))
        self.sense_groups.sort(key=lambda x: x.ratio, reverse=True)

        print('\nSenses groups', time.strftime("%H:%M:%S"))

        for sense_group in self.sense_groups:
            print(sense_group.ratio, sense_group.senses, sense_group.nouns)





def main():
    # reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review-short.json"
    # reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review.json"
    # reviews_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review_spas.json"
    reviews_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review_hotels.json"
    # reviews_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review_restaurants.json"
    reviews = context_utils.load_reviews(reviews_file)

    # If we want to use the sentences instead of the reviews to cluster then
    # we have to uncomment the following block
    # sentences = []
    # for review in context_utils.load_reviews(reviews_file):
    #     for sentence in tokenize.sent_tokenize(review):
    #         sentences.append(sentence)
    # reviews = [review for review in context_utils.load_reviews(reviews_file)]
    # print("sentences:", len(sentences))

    print("reviews:", len(reviews))

    # word_context = WordBasedContext(sentences)
    word_context = WordBasedContext(reviews)
    word_context.calculate_sense_group_ratios()

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
