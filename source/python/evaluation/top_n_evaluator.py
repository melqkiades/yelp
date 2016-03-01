import copy
import random
import cPickle as pickle

from etl import ETLUtils
from tripadvisor.fourcity import extractor

# Based on the code created by Osman Baskaya
from utils.constants import Constants

__author__ = 'fpena'


class TopNEvaluator:

    def __init__(self, records, test_records, item_type, N=10, I=1000):

        self.item_ids = None
        self.user_ids = None
        self.N = N
        self.I = I

        self.num_hits = 0
        self.num_misses = 0
        self.recall = 0
        self.precision = 0

        self.num_specific_hits = 0
        self.num_specific_misses = 0
        self.specific_recall = 0
        self.specific_precision = 0

        self.num_generic_hits = 0
        self.num_generic_misses = 0
        self.generic_recall = 0
        self.generic_precision = 0

        self.records = records
        self.training_set = None
        self.test_records = test_records
        self.item_type = item_type

        self.important_records = None
        self.items_to_predict = None
        self.records_to_predict = None
        self.user_item_map = None

    def initialize(self, user_item_map):
        self.user_ids =\
            extractor.get_groupby_list(self.records, Constants.USER_ID_FIELD)
        self.item_ids =\
            extractor.get_groupby_list(self.records, Constants.ITEM_ID_FIELD)
        print('total users', len(self.user_ids))
        print('total items', len(self.item_ids))
        self.user_item_map = user_item_map

        self.find_important_records()

    def get_irrelevant_items(self, user_id):
        user_items = self.user_item_map[user_id]
        diff_items = list(set(self.item_ids).difference(user_items))
        random.shuffle(diff_items)
        return diff_items

    def find_important_records(self):
        self.important_records = [
            record for record in self.test_records
            if record[Constants.RATING_FIELD] == 5]  # userItem is 5 rated film

    @staticmethod
    def create_top_n_list(rating_list, n):
        sorted_list = sorted(
            rating_list, key=rating_list.get, reverse=True)

        return sorted_list[:n]

    def calculate_precision(self):
        self.calculate_recall()
        self.precision = self.recall / self.N
        self.specific_precision = self.specific_recall / self.N
        self.generic_precision = self.generic_recall / self.N

    def calculate_recall(self):
        self.recall = float(self.num_hits) / (self.num_hits + self.num_misses)
        self.specific_recall = float(self.num_specific_hits) /\
            (self.num_specific_hits + self.num_specific_misses)
        self.generic_recall = float(self.num_generic_hits) /\
            (self.num_generic_hits + self.num_generic_misses)

    def evaluate_pr(self):
        self.calculate_recall()
        self.calculate_precision()
        return self.precision, self.recall

    def update_num_hits(self, top_n_list, item, review_type):

        if item in top_n_list:
            self.num_hits += 1
            if review_type == Constants.SPECIFIC:
                self.num_specific_hits += 1
            elif review_type == Constants.GENERIC:
                self.num_generic_hits += 1
            # print 'hit for item:%s\n' % item
        else:
            self.num_misses += 1
            if review_type == Constants.SPECIFIC:
                self.num_specific_misses += 1
            elif review_type == Constants.GENERIC:
                self.num_generic_misses += 1
            # print 'miss for item:%s\n' % item

    def get_records_to_predict(self):

        all_items_to_predict = {}
        all_records_to_predict = []

        for i in range(len(self.important_records)):
            record = self.important_records[i]

            user_id = record[Constants.USER_ID_FIELD]
            item_id = record[Constants.ITEM_ID_FIELD]
            # return I many of items
            irrelevant_items = self.get_irrelevant_items(user_id)[:self.I]

            if len(irrelevant_items) != self.I:
                print('Irrelevant items size is',
                      len(irrelevant_items), user_id, item_id)

            if irrelevant_items is not None:
                # add our relevant item for prediction
                irrelevant_items.append(item_id)
                user_item_key = user_id + '|' + item_id
                all_items_to_predict[user_item_key] = irrelevant_items

                for irrelevant_item in irrelevant_items:
                    generated_record = record.copy()
                    generated_record[Constants.ITEM_ID_FIELD] = irrelevant_item
                    all_records_to_predict.append(generated_record)

        self.items_to_predict = all_items_to_predict
        self.records_to_predict = all_records_to_predict

        return all_records_to_predict

    def export_records_to_predict(self, records_file):
        if self.records_to_predict is None:
            self.records_to_predict = self.get_records_to_predict()
        ETLUtils.save_json_file(records_file, self.records_to_predict)
        with open(records_file + '.pkl', 'wb') as write_file:
            pickle.dump(
                self.items_to_predict, write_file, pickle.HIGHEST_PROTOCOL)

    def load_records_to_predict(self, records_file):
        self.records_to_predict = ETLUtils.load_json_file(records_file)
        with open(records_file + '.pkl', 'rb') as read_file:
            self.items_to_predict = pickle.load(read_file)

    # def load_predictions(self, predictions_file):
    #     self.predictions =\
    #         rmse_calculator.read_targets_from_txt(predictions_file)

    def evaluate(self, predictions):

        print('num_important_items', len(self.important_records))
        print('num_predictions', len(predictions))
        print('I', self.I)
        assert len(predictions) == len(self.important_records) * (self.I + 1)

        index = 0
        self.num_hits = 0
        self.num_misses = 0
        self.num_specific_hits = 0
        self.num_specific_misses = 0
        self.num_generic_hits = 0
        self.num_generic_misses = 0

        for record in self.important_records:
            user_id = record[Constants.USER_ID_FIELD]
            item_id = record[Constants.ITEM_ID_FIELD]
            review_type = record[Constants.PREDICTED_CLASS_FIELD]
            user_item_key = user_id + '|' + item_id

            item_rating_map = {}
            irrelevant_items = self.items_to_predict[user_item_key]

            assert len(irrelevant_items) == self.I + 1
            for irrelevant_item in irrelevant_items:
                rating = predictions[index]
                item_rating_map[irrelevant_item] = rating

                index += 1

            top_n_list = self.create_top_n_list(item_rating_map, self.N)
            # use this inf. for calculating PR
            self.update_num_hits(top_n_list, item_id, review_type)

        self.calculate_precision()
        self.calculate_recall()

        # print('precision', self.precision)
        # print('recall', self.recall)


# start = time.time()
# # main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
