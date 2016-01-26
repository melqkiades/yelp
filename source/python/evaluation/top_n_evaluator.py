import random
import time
import cPickle as pickle
from subprocess import call

from etl import ETLUtils
from etl.libfm_converter import csv_to_libfm
from evaluation import rmse_calculator
from topicmodeling.context.lda_based_context import LdaBasedContext
from tripadvisor.fourcity import extractor

__author__ = 'Osman Baskaya & fpena'

my_i = 1000
my_dataset = 'hotel'
# my_dataset = 'restaurant'


class TopNEvaluator:

    def __init__(self, dataset, test, item_type, N=10, I=1000):

        self.item_ids = None
        self.user_ids = None
        self.N = N
        self.I = I

        self.n_hit = 0
        self.n_miss = 0
        self.recall = 0
        self.precision = 0

        self.dataset = dataset
        self.training_set = None
        self.test_set = test
        self.item_type = item_type

        self.important_items = None
        self.items_to_predict = None
        self.records_to_predict = None
        self.user_item_map = None

    def initialize(self):
        self.user_ids = extractor.get_groupby_list(self.dataset, 'user_id')
        self.item_ids = extractor.get_groupby_list(self.dataset, 'business_id')

        self.user_item_map = {}

        print('total users', len(self.user_ids))
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
        dataset = self.item_type
        my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
        output_file = my_folder + 'generated/' + dataset + '_user_item_map'
        # with open(output_file + '.pkl', 'wb') as write_file:
        #     pickle.dump(self.user_item_map, write_file, pickle.HIGHEST_PROTOCOL)

        with open(output_file + '.pkl', 'rb') as read_file:
            self.user_item_map = pickle.load(read_file)

        self.important_items =\
            TopNEvaluator.calculate_important_items(self.test_set)

    def get_irrelevant_items(self, user_id):
        user_items = self.user_item_map[user_id]
        diff_items = list(set(self.item_ids).difference(user_items))
        random.shuffle(diff_items)
        return diff_items

    @staticmethod
    def calculate_important_items(dataset):
        important_items = [record for record in dataset
                           if record['stars'] == 5]  # userItem is 5 rated film
        return important_items

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

        # print(self.important_items)

        for record in self.important_items:
            user_id = record['user_id']
            item_id = record['business_id']
            # return I many of items
            irrelevant_items = self.get_irrelevant_items(user_id)[:self.I]

            if len(irrelevant_items) != self.I:
                print('Irrelevan items size is', len(irrelevant_items), user_id, item_id)

            if irrelevant_items is not None:
                # add our relevant item for prediction
                irrelevant_items.append(item_id)
                user_item_key = user_id + '|' + item_id
                all_items_to_predict[user_item_key] = irrelevant_items

                for irrelevant_item in irrelevant_items:
                    generated_record = record.copy()
                    generated_record['business_id'] = irrelevant_item
                    all_records_to_predict.append(generated_record)

        self.items_to_predict = all_items_to_predict
        self.records_to_predict = all_records_to_predict

        return all_records_to_predict

    def export_records_to_predict(self, output_file):
        if self.records_to_predict is None:
            self.records_to_predict = self.get_records_to_predict()
        ETLUtils.save_json_file(output_file, self.records_to_predict)
        with open(output_file + '.pkl', 'wb') as write_file:
            pickle.dump(self.items_to_predict, write_file, pickle.HIGHEST_PROTOCOL)

    def load_records_to_predict(self, file_path):
        self.records_to_predict = ETLUtils.load_json_file(file_path)
        with open(file_path + '.pkl', 'rb') as read_file:
            self.items_to_predict = pickle.load(read_file)

    # def load_predictions(self, predictions_file):
    #     self.predictions =\
    #         rmse_calculator.read_targets_from_txt(predictions_file)

    def evaluate(self, predictions):

        # print('num_items', len(self.item_ids))
        print('num_important_items', len(self.important_items))
        print('num_predictions', len(predictions))
        print('I', self.I)
        assert len(predictions) == len(self.important_items) * (self.I + 1)

        # for record in self.important_items:
        #     print(record['user_id'], record['business_id'], record['stars'])

        # if self.items_to_predict is None:
        #     self.get_records_to_predict()

        index = 0
        # self.important_items = self.calculate_important_items(self.test_set)

        for record in self.important_items:
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










def main2(self, test_db, k=1, N=20, I=300):

    # pr = PrecisionRecall(self.dataset, test_db, N, I)
    pr = TopNEvaluator(self.trainset, test_db, N, I)
    rating_list = []
    # actual_testset = pr.calculate_important_items() # This item is high rated by users.
    # print 'Actual Length of the Testset%s\n' % actual_testset
    for user, item in actual_testset:
        #print "\nNew test for user_id=%s, item_id=%s\n" % (user, item)
        irrelevant_items = pr.get_irrelevant_items(user)

        if irrelevant_items is not None:

            irrelevant_items.append(item)  # added our relevant item for prediction
            assert len(irrelevant_items) == I + 1
            for item in irrelevant_items:
                key = str(user) + '_' + str(item)
                rating = self.rec.predict(user, item, signature=key)
                rating_list.append([item, rating])
            top_n_list = pr.create_top_n_list(rating_list, N)
            pr.update_num_hits(top_n_list, item)  # use this inf. for calculating PR

    precision, recall = pr.evaluate_pr()

    return precision, recall


def main3():

    dataset = my_dataset
    # dataset = 'hotel'
    # dataset = 'restaurant'
    I = my_i

    my_training_records_file = '/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'
    my_training_records = ETLUtils.load_json_file(my_training_records_file)

    train_records, test_records =\
            ETLUtils.split_train_test(my_training_records, split=0.8, shuffle_data=False)

    top_n_evaluator = TopNEvaluator(my_training_records, test_records)
    # top_n_evaluator.calculate_important_items()

    for record in top_n_evaluator.important_items:
        user_id = record['user_id']
        item_id = record['business_id']
        irrelevant_items = top_n_evaluator.get_irrelevant_items(user_id)

        if irrelevant_items is not None:
            irrelevant_items.append(item_id)  # added our relevant item for prediction
            assert len(irrelevant_items) == I + 1

            for irrelevant_item in irrelevant_items:
                pass


def main_export():
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    dataset = my_dataset
    # dataset = 'hotel'
    # dataset = 'restaurant'
    I = my_i

    my_records_file = my_folder + 'yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'
    my_records = ETLUtils.load_json_file(my_records_file)

    print('num_records', len(my_records))

    my_test_file = my_records_file + '_test'
    my_test_records = ETLUtils.load_json_file(my_test_file)

    top_n_evaluator = TopNEvaluator(my_records, my_test_records, 10, I)
    top_n_evaluator.initialize()
    # top_n_evaluator.get_records_to_predict()

    my_records_to_predict_file = my_folder + 'generated/records_to_predict_' + dataset + '.json'
    top_n_evaluator.export_records_to_predict(my_records_to_predict_file)


def main_load():
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    dataset = my_dataset
    # dataset = 'hotel'
    # dataset = 'restaurant'
    I = my_i

    my_records_file = my_folder + 'yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'
    my_records = ETLUtils.load_json_file(my_records_file)

    # print('num_records', len(my_records))

    my_test_file = my_records_file + '_test'
    my_test_records = ETLUtils.load_json_file(my_test_file)

    top_n_evaluator = TopNEvaluator(my_records, my_test_records, 10, I)
    top_n_evaluator.important_items =\
        TopNEvaluator.calculate_important_items(my_test_records)
    # top_n_evaluator.initialize()

    my_records_to_predict_file = my_folder + 'generated/records_to_predict_' + dataset + '.json'
    top_n_evaluator.load_records_to_predict(my_records_to_predict_file)

    my_predictions_file = '/Users/fpena/tmp/libfm-1.42.src/bin/predictions_' + dataset + '_no_context.txt'
    my_predictions = rmse_calculator.read_targets_from_txt(my_predictions_file)

    # print('total predictions', len(my_predictions))
    top_n_evaluator.evaluate(my_predictions)
    # print('precision', top_n_evaluator.precision)
    print('recall', top_n_evaluator.recall)

    return top_n_evaluator.recall


def main_converter():

    # my_input_folder = '/Users/fpena/tmp/libfm-1.42.src/bin/'
    my_input_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    dataset = my_dataset
    my_records_file = my_input_folder + 'yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'
    json_file1 = my_records_file + '_train'
    json_file2 = my_input_folder + 'generated/' + 'records_to_predict_hotel.json'

    my_export_folder = '/Users/fpena/tmp/libfm-1.42.src/bin/'
    my_export_file = my_export_folder + 'yelp_training_set_review_hotels_shuffled_train.csv'
    # csv_file = my_export_folder + 'yelp3.csv'
    csv_file = my_export_folder + 'records_to_predict_hotel.csv'
    libfm_file = my_export_folder + 'yelp_delete.libfm'

    ETLUtils.json_to_csv(json_file1, my_export_file, 'user_id', 'business_id', 'stars', False, True)
    ETLUtils.json_to_csv(json_file2, csv_file, 'user_id', 'business_id', 'stars', False, True)

    csv_files = [
        my_export_file,
        csv_file
    ]

    csv_to_libfm(csv_files, 2, [0, 1], [], ',', has_header=True)


def main_libfm():

    folder = '/Users/fpena/tmp/libfm-1.42.src/bin/'
    libfm_command = folder + 'libfm'
    train_file = folder + 'yelp_training_set_review_hotels_shuffled_train.csv.libfm'
    test_file = folder + 'records_to_predict_hotel.csv.libfm'
    predictions_file = folder + 'predictions_hotel_no_context.txt'
    log_file = folder + 'hotel_no_context.log'

    command = [
        libfm_command,
        '-task',
        'r',
        '-train',
        train_file,
        '-test',
        test_file,
        '-dim',
        '1,1,8',
        '-out',
        predictions_file
        # '>',
        # log_file
    ]

    # libFM -task r -train /Users/fpena/tmp/libfm-1.42.src/bin/yelp_training_set_review_hotels_shuffled_train.csv.libfm
    # -test /Users/fpena/tmp/libfm-1.42.src/bin/records_to_predict_hotel.csv.libfm
    # -dim '1,1,8' -out /Users/fpena/tmp/libfm-1.42.src/bin/predictions_hotel_no_context.txt > /Users/fpena/tmp/libfm-1.42.src/bin/hotel_no_context.txt

    f = open(log_file, "w")
    call(command, stdout=f)


def main():
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    dataset = my_dataset
    # dataset = 'hotel'
    # dataset = 'restaurant'
    I = my_i

    my_records_file = my_folder + 'yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'
    my_records = ETLUtils.load_json_file(my_records_file)

    print('num_records', len(my_records))

    my_test_file = my_folder + my_records_file + '_test'
    my_test_records = ETLUtils.load_json_file(my_test_file)

    top_n_evaluator = TopNEvaluator(my_records, my_test_records, 10, I)
    top_n_evaluator.initialize()

    # my_records_to_predict = top_n_evaluator.get_records_to_predict()
    my_records_to_predict_file = my_folder + 'generated/records_to_predict_' + dataset + '.json'
    # top_n_evaluator.export_records_to_predict(my_records_to_predict_file)
    top_n_evaluator.load_records_to_predict(my_records_to_predict_file)

    # for record in my_records_to_predict:
    #     print(record)

    # # my_predictions_file = '/Users/fpena/tmp/libfm-1.42.src/bin/predictions_hotel_context.txt'
    my_predictions_file = '/Users/fpena/tmp/libfm-1.42.src/bin/predictions_hotel_no_context.txt'
    my_predictions = rmse_calculator.read_targets_from_txt(my_predictions_file)
    #
    # print(my_predictions)
    # print('total predictions', len(my_predictions))

    # train_records, test_records =\
    #     ETLUtils.split_train_test(my_records, split=0.8, shuffle_data=False)


    # top_n_evaluator.calculate_important_items()
    top_n_evaluator.evaluate(my_predictions)
    # print('precision', top_n_evaluator.precision)
    print('recall', top_n_evaluator.recall)


def main_split():

    dataset = my_dataset
    split_percentage = '98'
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    my_records_file = my_folder + 'yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'

    split_command = my_folder + 'split_file.sh'

    command = [
        split_command,
        my_records_file,
        my_records_file,
        split_percentage
    ]

    call(command)

def main_lda():

    dataset = my_dataset

    my_training_records_file = '/Users/fpena/UCC/Thesis/datasets/context/yelp_training_set_review_' + dataset + 's_shuffled_tagged.json'
    my_training_records = ETLUtils.load_json_file(my_training_records_file)
    my_training_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/reviews_' + dataset + '_shuffled.pkl'
    # my_reviews = context_utils.load_reviews(reviews_file)
    # print("reviews:", len(my_reviews))
    #
    # my_reviews = None
    # my_file = '/Users/fpena/tmp/reviews_restaurant_shuffled.pkl'
    # my_file = '/Users/fpena/tmp/sentences_hotel.pkl'
    # my_file = '/Users/fpena/tmp/reviews_hotel.pkl'
    # my_file = '/Users/fpena/tmp/reviews_spa.pkl'

    # with open(my_file, 'wb') as write_file:
    #     pickle.dump(self.reviews, write_file, pickle.HIGHEST_PROTOCOL)
    # training_records_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_' + dataset + '_reviews.json'
    # training_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_' + dataset + '_reviews.pkl'

    with open(my_training_reviews_file, 'rb') as read_file:
        my_training_reviews = pickle.load(read_file)

    print('lda num_reviews', len(my_training_reviews))
    # lda_context_utils.discover_topics(my_reviews, 150)
    lda_based_context = LdaBasedContext(my_training_records, my_training_reviews)
    # lda_based_context.training_set_file = training_records_file
    # lda_based_context.training_reviews_file = training_reviews_file
    lda_based_context.init_reviews()

    my_topics = lda_based_context.get_context_rich_topics()
    print(my_topics)
    print('total_topics', len(my_topics))

    my_records_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_' + dataset + '_reviews.json'
    my_reviews_file = '/Users/fpena/UCC/Thesis/datasets/context/classified_' + dataset + '_reviews.pkl'
    json_file = '/Users/fpena/UCC/Thesis/datasets/context/yelp_' + dataset + '_context_shuffled4.json'
    csv_file = '/Users/fpena/UCC/Thesis/datasets/context/yelp_' + dataset + '_context_shuffled4.csv'
    lda_based_context.export_contextual_records(my_records_file, my_reviews_file, json_file, csv_file)



def super_main():

    total_recall = 0.0
    num_iterations = 10

    for _ in range(num_iterations):
        print('main split')
        main_split()
        print('main export')
        main_export()
        print('main converter')
        main_converter()
        print('main libfm')
        main_libfm()
        print('main load')
        total_recall += main_load()

    average_recall = total_recall / num_iterations
    print('average_recall', average_recall)

# start = time.time()
# # main()
# # main_split()
# # main_export()
# # main_converter()
# # main_libfm()
# # main_load()
# # super_main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)


