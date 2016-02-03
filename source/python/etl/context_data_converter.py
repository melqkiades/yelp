import time

from etl import ETLUtils
from etl import libfm_converter
from evaluation.top_n_evaluator import TopNEvaluator
from topicmodeling.context.lda_based_context import LdaBasedContext

__author__ = 'fpena'


class ContextDataConverter:

    def __init__(self, reviews_classifier):
        self.reviews_classifier = reviews_classifier
        self.shuffle_seed = 0
        # self.user_ids = None
        # self.item_ids = None
        # self.num_users = None
        # self.num_items = None
        self.headers = None
        self.lda_based_context = None

    def full_cycle(
            self, train_records, test_records, train_reviews, test_reviews):

        # self.user_ids = extractor.get_groupby_list(records, 'user_id')
        # self.item_ids = extractor.get_groupby_list(records, 'business_id')
        # self.num_users = len(self.user_ids)
        # self.num_items = len(self.item_ids)
        # print('Preloaded users and items: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        # train_records = self.reviews_classifier.label_json_reviews(train_records, train_reviews)
        # test_records = self.reviews_classifier.label_json_reviews(test_records, test_reviews)

        print('Classified records: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        # if other_rated_items:
        #     records = self.add_other_rated_items_info(records)
        # train_records, test_records =\
        #     ETLUtils.split_train_test(records, split=0.8, shuffle_data=False)
        # train_reviews, test_reviews =\
        #     ETLUtils.split_train_test(reviews, split=0.8, shuffle_data=False)

        # print('Splitted data: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        # specific_test_records, specific_test_reviews =\
        #     zip(*((record, review)
        #           for record, review in zip(test_records, test_reviews)
        #           if record['predicted_class'] in ['specific']))
        #
        # generic_test_records, generic_test_reviews =\
        #     zip(*((record, review)
        #           for record, review in zip(test_records, test_reviews)
        #           if record['predicted_class'] in ['generic']))

        print('Filtered data: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        self.lda_based_context = LdaBasedContext(train_records, train_reviews)
        self.lda_based_context.get_context_rich_topics()

        print('Trained LDA Model: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        contextual_train_set =\
            self.lda_based_context.find_contextual_topics(
                train_records, train_reviews)
        contextual_test_set = self.lda_based_context.find_contextual_topics(
            test_records, test_reviews)

        print('contextual test set size: %d' % len(contextual_test_set))

        # specific_contextual_test_set =\
        #     self.lda_based_context.find_contextual_topics(
        #         specific_test_records, test_reviews)
        # generic_contextual_test_set =\
        #     self.lda_based_context.find_contextual_topics(
        #         generic_test_records, test_reviews)

        self.build_headers()
        contextual_train_set =\
            ETLUtils.select_fields(self.headers, contextual_train_set)
        contextual_test_set =\
            ETLUtils.select_fields(self.headers, contextual_test_set)
        # specific_contextual_test_set =\
        #     ETLUtils.select_fields(self.headers, specific_contextual_test_set)
        # generic_contextual_test_set =\
        #     ETLUtils.select_fields(self.headers, generic_contextual_test_set)

        print('Exported contextual topics: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        return contextual_train_set, contextual_test_set, None, None
        # return contextual_train_set, contextual_test_set, \
            # specific_contextual_test_set, generic_contextual_test_set

    def build_headers(self):
        self.headers = [TopNEvaluator.RATING_FIELD, TopNEvaluator.USER_ID_FIELD, TopNEvaluator.ITEM_ID_FIELD]
        # if other_rated_items:
        #     for item_id in self.item_ids:
        #         self.headers.append('rated_' + item_id)
        for i in self.lda_based_context.context_rich_topics:
            topic_id = 'topic' + str(i[0])
            self.headers.append(topic_id)

    def run(self, dataset, output_folder, train_records, test_records,
            train_reviews=None, test_reviews=None):

        # other_items = False

        contextual_train_set, contextual_test_set,\
            specific_contextual_test_set, generic_contextual_test_set =\
            self.full_cycle(
                train_records, test_records,
                train_reviews, test_reviews)

        print('Prepared data: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        # json_train_file = output_folder + 'yelp_' + dataset + '_context_shuffled_train5.json'
        csv_train_file = output_folder + 'yelp_' + dataset + '_context_shuffled_train5.csv'
        # json_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_test5.json'
        csv_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_test5.csv'
        # specific_json_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_specific_test5.json'
        # specific_csv_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_specific_test5.csv'
        # generic_json_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_generic_test5.json'
        # generic_csv_test_file = output_folder + 'yelp_' + dataset + '_context_shuffled_generic_test5.csv'

        # ETLUtils.save_json_file(json_train_file, contextual_train_set)
        ETLUtils.save_csv_file(csv_train_file, contextual_train_set, self.headers)

        # ETLUtils.save_json_file(json_test_file, contextual_test_set)
        ETLUtils.save_csv_file(csv_test_file, contextual_test_set, self.headers)
        # ETLUtils.save_json_file(specific_json_test_file, specific_contextual_test_set)
        # ETLUtils.save_csv_file(specific_csv_test_file, specific_contextual_test_set, self.headers)
        # ETLUtils.save_json_file(generic_json_test_file, generic_contextual_test_set)
        # ETLUtils.save_csv_file(generic_csv_test_file, generic_contextual_test_set, self.headers)

        print('Exported CSV and JSON files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))

        csv_files = [
            csv_train_file,
            csv_test_file,
            # specific_csv_test_file,
            # generic_csv_test_file
        ]

        num_cols = len(self.headers)
        context_cols = num_cols
        # if other_items:
        #     context_cols = num_cols - self.num_items
        print('num_cols', num_cols)
        # print('context_cols', context_cols)

        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], range(3, context_cols), ',', has_header=True,
            suffix='.no_context.libfm')
        libfm_converter.csv_to_libfm(
            csv_files, 0, [1, 2], [], ',', has_header=True,
            suffix='.context.libfm')

        print('Exported LibFM files: %s' % time.strftime("%Y/%d/%m-%H:%M:%S"))


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
