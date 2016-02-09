import time

from etl import ETLUtils
from etl import libfm_converter
from topicmodeling.context.lda_based_context import LdaBasedContext

__author__ = 'fpena'


class ContextDataConverter:

    def __init__(self, reviews_classifier):
        self.reviews_classifier = reviews_classifier
        self.shuffle_seed = 0
        self.headers = None
        self.lda_based_context = None

    def full_cycle(
            self, train_records, test_records, train_reviews, test_reviews):

        self.lda_based_context = LdaBasedContext(train_records, train_reviews)
        self.lda_based_context.get_context_rich_topics()

        print('Trained LDA Model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        contextual_train_set =\
            self.lda_based_context.find_contextual_topics(train_records)
        contextual_test_set =\
            self.lda_based_context.find_contextual_topics(test_records)

        print('contextual test set size: %d' % len(contextual_test_set))

        self.build_headers()
        contextual_train_set =\
            ETLUtils.select_fields(self.headers, contextual_train_set)
        contextual_test_set =\
            ETLUtils.select_fields(self.headers, contextual_test_set)

        print('Exported contextual topics: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        return contextual_train_set, contextual_test_set

    def build_headers(self):
        self.headers = ['stars', 'user_id', 'business_id']
        for i in self.lda_based_context.context_rich_topics:
            topic_id = 'topic' + str(i[0])
            self.headers.append(topic_id)

    def run(self, dataset, output_folder, train_records, test_records,
            train_reviews=None, test_reviews=None):

        contextual_train_set, contextual_test_set =\
            self.full_cycle(
                train_records, test_records,
                train_reviews, test_reviews)

        print('Prepared data: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        # json_train_file = output_folder + 'yelp_' + dataset + '_context_shuffled_train5.json'
        csv_train_file = output_folder + 'yelp_' + dataset + '_context_shuffled_train5.csv'
        # json_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_test5.json'
        csv_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_test5.csv'

        # ETLUtils.save_json_file(json_train_file, contextual_train_set)
        ETLUtils.save_csv_file(csv_train_file, contextual_train_set, self.headers)

        # ETLUtils.save_json_file(json_test_file, contextual_test_set)
        ETLUtils.save_csv_file(csv_test_file, contextual_test_set, self.headers)

        print('Exported CSV and JSON files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        csv_files = [
            csv_train_file,
            csv_test_file
        ]

        num_cols = len(self.headers)
        context_cols = num_cols
        print('num_cols', num_cols)
        # print('context_cols', context_cols)

        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], range(3, context_cols), ',', has_header=True,
            suffix='.no_context.libfm')
        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], [], ',', has_header=True,
            suffix='.context.libfm')

        print('Exported LibFM files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

def get_rated_items(records, user):

    return [
        record['business_id']
        for record in records if record['user_id'] == user]


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
