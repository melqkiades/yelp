import cPickle as pickle
import random
import time
from etl import ETLUtils
from etl import libfm_converter
from topicmodeling.context import review_metrics_extractor
from topicmodeling.context.lda_based_context import LdaBasedContext
from topicmodeling.context.reviews_classifier import ReviewsClassifier
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class ContextDataConverter:

    def __init__(self, reviews_classifier):
        self.reviews_classifier = reviews_classifier
        self.shuffle_seed = 0
        self.user_ids = None
        self.item_ids = None
        self.num_users = None
        self.num_items = None
        self.headers = None
        self.lda_based_context = None

    def full_cycle(
            self, json_file, reviews_file=None, shuffle_data=False,
            other_rated_items=False):

        records = ETLUtils.load_json_file(json_file)

        print('Loaded records: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        if reviews_file is not None:
            with open(reviews_file, 'rb') as read_file:
                reviews = pickle.load(read_file)
        else:
            reviews = review_metrics_extractor.build_reviews(records)

        print('Loaded reviews: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        self.user_ids = extractor.get_groupby_list(records, 'user_id')
        self.item_ids = extractor.get_groupby_list(records, 'business_id')
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

        print('Preloaded users and items: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        if shuffle_data:
            # By using a seed we maintain the same order after shuffling the
            # lists
            random.seed(self.shuffle_seed)
            random.shuffle(records)
            random.seed(self.shuffle_seed)
            random.shuffle(reviews)

        records = self.reviews_classifier.label_json_reviews(records, reviews)

        print('Classified records: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        if other_rated_items:
            records = self.add_other_rated_items_info(records)
        train_records, test_records =\
            ETLUtils.split_train_test(records, split=0.8, shuffle_data=False)
        train_reviews, test_reviews =\
            ETLUtils.split_train_test(reviews, split=0.8, shuffle_data=False)

        print('Splitted data: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        specific_test_records, specific_test_reviews =\
            zip(*((record, review)
                  for record, review in zip(test_records, test_reviews)
                  if record['predicted_class'] in ['specific']))

        generic_test_records, generic_test_reviews =\
            zip(*((record, review)
                  for record, review in zip(test_records, test_reviews)
                  if record['predicted_class'] in ['generic']))

        print('Filtered data: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        self.lda_based_context = LdaBasedContext(train_records, train_reviews)
        self.lda_based_context.get_context_rich_topics()

        print('Trained LDA Model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        contextual_train_set =\
            self.lda_based_context.find_contextual_topics(
                train_records, train_reviews)
        contextual_test_set = self.lda_based_context.find_contextual_topics(
            test_records, test_reviews)
        specific_contextual_test_set =\
            self.lda_based_context.find_contextual_topics(
                specific_test_records, test_reviews)
        generic_contextual_test_set =\
            self.lda_based_context.find_contextual_topics(
                generic_test_records, test_reviews)

        self.build_headers(other_rated_items)

        # We drop unnecessary fields
        for record in records:
            record['rating'] = record.pop('stars')
            record['item_id'] = record.pop('business_id')

        contextual_train_set =\
            ETLUtils.select_fields(self.headers, contextual_train_set)
        contextual_test_set =\
            ETLUtils.select_fields(self.headers, contextual_test_set)
        specific_contextual_test_set =\
            ETLUtils.select_fields(self.headers, specific_contextual_test_set)
        generic_contextual_test_set =\
            ETLUtils.select_fields(self.headers, generic_contextual_test_set)

        print('Exported contextual topics: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        return contextual_train_set, contextual_test_set,\
            specific_contextual_test_set, generic_contextual_test_set

    def build_headers(self, other_rated_items):
        self.headers = ['rating', 'user_id', 'item_id']
        if other_rated_items:
            for item_id in self.item_ids:
                self.headers.append('rated_' + item_id)
        for i in self.lda_based_context.context_rich_topics:
            topic_id = 'topic' + str(i[0])
            self.headers.append(topic_id)

    def run(self, dataset):

        input_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
        # output_folder = input_folder + 'generated/'
        output_folder = '/Users/fpena/tmp/libfm-1.42.src/bin/'

        my_records_file = input_folder + 'yelp_training_set_review_' + dataset + 's_shuffled.json'
        my_reviews_file = input_folder + 'reviews_' + dataset + '_shuffled.pkl'

        other_items = False

        contextual_train_set, contextual_test_set,\
            specific_contextual_test_set, generic_contextual_test_set =\
            self.full_cycle(
                my_records_file, my_reviews_file, other_rated_items=other_items)

        print('Prepared data: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        # json_train_file = output_folder + 'yelp_' + dataset + '_context_shuffled_train5.json'
        csv_train_file = output_folder + 'yelp_' + dataset + '_context_shuffled_train5.csv'
        # json_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_test5.json'
        csv_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_test5.csv'
        # specific_json_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_specific_test5.json'
        specific_csv_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_specific_test5.csv'
        # generic_json_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_generic_test5.json'
        generic_csv_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_generic_test5.csv'

        # ETLUtils.save_json_file(json_train_file, contextual_train_set)
        ETLUtils.save_csv_file(csv_train_file, contextual_train_set, self.headers)

        # ETLUtils.save_json_file(json_test_file, contextual_test_set)
        ETLUtils.save_csv_file(csv_test_file, contextual_test_set, self.headers)
        # ETLUtils.save_json_file(specific_json_test_file, specific_contextual_test_set)
        ETLUtils.save_csv_file(specific_csv_test_file, specific_contextual_test_set, self.headers)
        # ETLUtils.save_json_file(generic_json_test_file, generic_contextual_test_set)
        ETLUtils.save_csv_file(generic_csv_test_file, generic_contextual_test_set, self.headers)

        print('Exported CSV and JSON files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        csv_files = [
            csv_train_file,
            csv_test_file,
            specific_csv_test_file,
            generic_csv_test_file
        ]

        num_cols = len(self.headers)
        context_cols = num_cols
        if other_items:
            context_cols = num_cols - self.num_items
        print('num_cols', num_cols)
        print('context_cols', context_cols)

        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], range(3, context_cols), ',', has_header=True,
            suffix='.no_context.libfm')
        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], [], ',', has_header=True,
            suffix='.context.libfm')

        print('Exported LibFM files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    def add_other_rated_items_info(self, records):

        user_items_map = {}
        for user_id in self.user_ids:
            user_items_map[user_id] = get_rated_items(records, user_id)

        for record in records:
            for item_id in self.item_ids:
                record['rated_' + item_id] = 0.0
            user_id = record['user_id']
            rated_items = user_items_map[user_id]
            fraction = 1.0/len(rated_items)

            for rated_item in rated_items:
                record['rated_' + rated_item] = fraction

        return records


def get_rated_items(records, user):

    return [
        record['business_id']
        for record in records if record['user_id'] == user]

def main():

    dataset = 'hotel'
    # dataset = 'restaurant'

    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'

    my_tagged_records_file = my_folder + 'classified_' + dataset + '_reviews.json'
    my_tagged_reviews_file = my_folder + 'classified_' + dataset + '_reviews.pkl'

    my_tagged_records = ETLUtils.load_json_file(my_tagged_records_file)
    with open(my_tagged_reviews_file, 'rb') as read_file:
        my_tagged_reviews = pickle.load(read_file)

    my_reviews_classifier = ReviewsClassifier()
    my_reviews_classifier.train(my_tagged_records, my_tagged_reviews)

    print('Trained classifier: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    my_data_preparer = ContextDataConverter(my_reviews_classifier)
    my_data_preparer.run(dataset)


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)


# dataset = 'hotel'
# # dataset = 'restaurant'
# my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
# my_records_file = my_folder + 'yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'
# my_reviews_file = my_folder + 'reviews_' + dataset + '_shuffled.pkl'
# my_records = ETLUtils.load_json_file(my_records_file)
# with open(my_reviews_file, 'rb') as read_file:
#     my_reviews = pickle.load(read_file)


# for record in my_records[:10]:
#     print(record)

# my_specific_records, my_specific_reviews = zip(*((record, review) for record, review in zip(my_records, my_reviews) if record['predicted_class'] in ['generic']))

# print(my_specific_records)
# my_specific_records = [record for record in my_records if record['predicted_class'] == 'specific']
# my_generic_records = [record for record in my_records if record['predicted_class'] == 'generic']
#
# #
# for record in my_specific_records[:10]:
#     print(record)
#
# print('\n\n\n******************\n\n\n')
#
# for record in my_generic_records[:10]:
#     print(record)
#
# updated_records = add_other_rated_items_info(my_records)
# for record in my_records:
#     print(record)



#
# print(len(my_specific_records))
# print(len(my_specific_reviews))
#
# specific_test_records, specific_test_reviews =\
#     zip(*((my_records, test_reviews)
#           for record, test_reviews in zip(my_records, my_reviews)
#           if record['predicted_class'] in ['specific']))
#
# # print(len(specific_test_records))
#
# # for record, review in zip(specific_test_records, specific_test_reviews)[:10]:
# #     print(record)
#     # print(review.text)
#     # print()
#     # print()





