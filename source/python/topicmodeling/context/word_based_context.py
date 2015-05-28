import time
from topicmodeling.context import context_utils
from topicmodeling.context.review import Review

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

        for text_review in self.text_reviews:
            self.reviews.append(Review(text_review))

        text_specific_reviews, text_generic_reviews =\
            context_utils.cluster_reviews(self.text_reviews)

        for text_review in text_specific_reviews:
            self.specific_reviews.append(Review(text_review))
        for text_review in text_generic_reviews:
            self.generic_reviews.append(Review(text_review))

        self.all_nouns = context_utils.get_all_nouns(self.reviews)

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
            print(sense_group.senses, sense_group.ratio)





def main():
    # reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review-short.json"
    reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review.json"
    reviews = context_utils.load_reviews(reviews_file)[:5000]
    print("reviews:", len(reviews))
    # specific, generic = context_utils.cluster_reviews(reviews)

    # print(specific)
    # print(generic)

    word_context = WordBasedContext(reviews)
    word_context.calculate_sense_group_ratios()

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
