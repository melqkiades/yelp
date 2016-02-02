import copy
import os
import random
import cPickle as pickle

from etl import ETLUtils
from tripadvisor.fourcity import extractor

# Based on the code created by Osman Baskaya
__author__ = 'fpena'


class TopNEvaluator:

    def __init__(self, records, test_records, item_type, N=10, I=1000,
                 test_reviews=None):

        self.item_ids = None
        self.user_ids = None
        self.N = N
        self.I = I

        self.n_hit = 0
        self.n_miss = 0
        self.recall = 0
        self.precision = 0

        self.records = records
        self.training_set = None
        self.test_records = test_records
        self.item_type = item_type

        self.important_records = None
        self.items_to_predict = None
        self.records_to_predict = None
        self.user_item_map = None

        self.test_reviews = test_reviews
        self.important_reviews = None
        self.reviews_to_predict = None

    def initialize(self, user_item_map):
        self.user_ids = extractor.get_groupby_list(self.records, 'user_id')
        self.item_ids = extractor.get_groupby_list(self.records, 'business_id')
        print('total users', len(self.user_ids))
        print('total items', len(self.item_ids))

        # self.user_item_map = {}
        #

        # user_count = 0
        #
        # for user_id in self.user_ids:
        #     user_records =\
        #         ETLUtils.filter_records(self.dataset, 'user_id', [user_id])
        #     user_items = extractor.get_groupby_list(user_records, 'business_id')
        #     self.user_item_map[user_id] = user_items
        #     user_count += 1
        #
        #     # print("user count %d" % user_count),
        #     print 'user count: {0}\r'.format(user_count),
        #
        # print
        # dataset = self.item_type
        # my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
        # output_file = my_folder + 'generated2/' + dataset + '_user_item_map.pkl'
        # with open(output_file + '.pkl', 'wb') as write_file:
        #     pickle.dump(self.user_item_map, write_file, pickle.HIGHEST_PROTOCOL)
        #
        # with open(output_file, 'rb') as read_file:
        #     self.user_item_map = pickle.load(read_file)
        self.user_item_map = user_item_map

        self.calculate_important_items()

    def get_irrelevant_items(self, user_id):
        user_items = self.user_item_map[user_id]
        diff_items = list(set(self.item_ids).difference(user_items))
        random.shuffle(diff_items)
        return diff_items

    def calculate_important_items(self):
        self.important_records = [
            record for record in self.test_records
            if record['stars'] == 5]  # userItem is 5 rated film
        if self.test_reviews is not None:
            self.important_reviews = [
                review for review in self.test_reviews
                if review.rating == 5]

    @staticmethod
    def create_top_n_list(rating_list, n):
        sorted_list = sorted(
            rating_list, key=rating_list.get, reverse=True)
        # sorted_list = sorted(
        #     rating_list, key=operator.itemgetter('stars'), reverse=True)

        return sorted_list[:n]

        # rating_array = numpy.array(rating_list)
        # I = numpy.argsort(rating_array[:, 1])
        # top_n_list = rating_array[I, :]
        # return top_n_list[-n:]  # negative. This is asc. We need high values

    def calculate_precision(self):
        return self.calculate_recall() / self.N

    def calculate_recall(self):
        return float(self.n_hit) / (self.n_hit + self.n_miss)

    def evaluate_pr(self):
        self.calculate_recall()
        self.calculate_precision()
        return self.precision, self.recall

    def update_num_hits(self, top_n_list, item):

        if item in top_n_list:
            self.n_hit += 1
            # print 'hit for item:%s\n' % item
        else:
            self.n_miss += 1
            # print 'miss for item:%s\n' % item

    def get_records_to_predict(self):

        all_items_to_predict = {}
        all_records_to_predict = []
        all_reviews_to_predict = []

        # print(self.important_items)

        # for record in self.important_items:
        for i in range(len(self.important_records)):
            record = self.important_records[i]

            review = None
            if self.important_reviews is not None:
                review = self.important_reviews[i]

            user_id = record['user_id']
            item_id = record['business_id']
            # return I many of items
            irrelevant_items = self.get_irrelevant_items(user_id)[:self.I]

            if len(irrelevant_items) != self.I:
                print('Irrelevant items size is', len(irrelevant_items), user_id, item_id)

            if irrelevant_items is not None:
                # add our relevant item for prediction
                irrelevant_items.append(item_id)
                user_item_key = user_id + '|' + item_id
                all_items_to_predict[user_item_key] = irrelevant_items

                for irrelevant_item in irrelevant_items:
                    generated_record = record.copy()
                    generated_record['business_id'] = irrelevant_item
                    all_records_to_predict.append(generated_record)

                    if review is not None:
                        generated_review = copy.copy(review)
                        generated_review.item_id = irrelevant_item
                        all_reviews_to_predict.append(generated_review)

        self.items_to_predict = all_items_to_predict
        self.records_to_predict = all_records_to_predict

        if self.important_reviews is not None:
            self.reviews_to_predict = all_reviews_to_predict

        return all_records_to_predict

    def export_records_to_predict(self, records_file, reviews_file=None):
        if self.records_to_predict is None:
            self.records_to_predict = self.get_records_to_predict()
        ETLUtils.save_json_file(records_file, self.records_to_predict)
        with open(records_file + '.pkl', 'wb') as write_file:
            pickle.dump(self.items_to_predict, write_file, pickle.HIGHEST_PROTOCOL)
        if reviews_file is not None:
            with open(reviews_file, 'wb') as write_file:
                pickle.dump(self.reviews_to_predict, write_file, pickle.HIGHEST_PROTOCOL)

    def load_records_to_predict(self, records_file, reviews_file=None):
        self.records_to_predict = ETLUtils.load_json_file(records_file)
        with open(records_file + '.pkl', 'rb') as read_file:
            self.items_to_predict = pickle.load(read_file)
        if reviews_file is not None:
            with open(reviews_file, 'rb') as read_file:
                self.reviews_to_predict = pickle.load(read_file)

    # def load_predictions(self, predictions_file):
    #     self.predictions =\
    #         rmse_calculator.read_targets_from_txt(predictions_file)

    def evaluate(self, predictions):

        # print('num_items', len(self.item_ids))
        print('num_important_items', len(self.important_records))
        print('num_predictions', len(predictions))
        print('I', self.I)
        assert len(predictions) == len(self.important_records) * (self.I + 1)

        # for record in self.important_items:
        #     print(record['user_id'], record['business_id'], record['stars'])

        # if self.items_to_predict is None:
        #     self.get_records_to_predict()

        index = 0
        # self.important_items = self.calculate_important_items(self.test_set)

        for record in self.important_records:
            user_id = record['user_id']
            item_id = record['business_id']
            user_item_key = user_id + '|' + item_id

            item_rating_map = {}
            irrelevant_items = self.items_to_predict[user_item_key]

            assert len(irrelevant_items) == self.I + 1
            for irrelevant_item in irrelevant_items:
                # key = str(user_id) + '|' + str(item_id)
                rating = predictions[index]
                item_rating_map[irrelevant_item] = rating

                index += 1

            top_n_list = self.create_top_n_list(item_rating_map, self.N)
            # use this inf. for calculating PR
            self.update_num_hits(top_n_list, item_id)

        self.precision = self.calculate_precision()
        self.recall = self.calculate_recall()

        # print('precision', self.precision)
        # print('recall', self.recall)


# start = time.time()
# # main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
