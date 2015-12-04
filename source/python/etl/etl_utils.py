import csv
import json
import random
import nltk
import numpy as np
from pandas import DataFrame

__author__ = 'franpena'


class ETLUtils:
    def __init__(self):
        pass

    @staticmethod
    def load_json_file(file_path):
        """
        Builds a list of dictionaries from a JSON file

        :type file_path: string
        :param file_path: the path for the file that contains the businesses
        data
        :return: a list of dictionaries with the data from the files
        """
        records = [json.loads(line) for line in open(file_path)]

        return records

    @staticmethod
    def save_json_file(file_path, records):
        with open(file_path, 'w') as outfile:
            for record in records:
                json.dump(record, outfile)
                outfile.write('\n')

    @staticmethod
    def drop_fields(fields, dictionary_list):
        """
        Removes the specified fields from every dictionary in the dictionary
        list

        :rtype : void
        :param fields: a list of     strings, which contains the fields that are
        going to be removed from every dictionary in the dictionary list
        :param dictionary_list: a list of dictionaries
        """
        for dictionary in dictionary_list:
            for field in fields:
                del (dictionary[field])

    @staticmethod
    def select_fields(fields, dictionary_list):
        """
        Returns a list of dictionaries with each dictionary containing only the
        keys given in the fields list

        :param fields: a list of the keys that each dictionary will have.
        This list must be a subset of all the keys available in each dictionary
        :param dictionary_list: a list of dictionaries
        :return: a list of dictionaries with each dictionary containing only the
        keys given in the fields list
        """
        filtered_records = [{field: dictionary[field] for field in fields} for
                            dictionary in dictionary_list]
        return filtered_records

    @staticmethod
    def filter_records(dictionary_list, field, values):
        """
        Returns a list with the dictionaries in dictionary_list that contain any
        of the values inside the field key. This method is the equivalent of
        SELECT * FROM my_table WHERE field IN (values) in SQL

        :param dictionary_list: a list of dictionaries
        :param field: the key of the dictionaries that is going to be used for
        filtering
        :param values: a list of values
        :return: a list with the dictionaries in dictionary_list that contain
        any of the values inside the field key
        """
        filtered_records = [dictionary for dictionary in dictionary_list if
                            dictionary[field] in values]
        return filtered_records

    @staticmethod
    def filter_out_records(dictionary_list, field, values):
        """
        Returns a list with the dictionaries in dictionary_list that do not
        contain any of the values inside the field key. This method is the
        equivalent of SELECT * FROM my_table WHERE field NOT IN (values) in SQL

        :param dictionary_list: a list of dictionaries
        :param field: the key of the dictionaries that is going to be used for
        filtering
        :param values: a list of values
        :return: a list with the dictionaries in dictionary_list that do not
        contain any of the values inside the field key
        """
        filtered_records = [dictionary for dictionary in dictionary_list if
                            dictionary[field] not in values]
        return filtered_records

    @staticmethod
    def add_transpose_list_column(field, dictionary_list):
        """
        Takes a list of dictionaries and adds to every dictionary a new field
        for each value contained in the specified field among all the
        dictionaries in the field, leaving 1 for the values that are present in
        the dictionary and 0 for the values that are not. It can be seen as
        transposing the dictionary matrix.

        :param field: the field which is going to be transposed
        :param dictionary_list: a list of dictionaries
        :return: the modified list of dictionaries
        """
        values_set = set()
        for dictionary in dictionary_list:
            values_set |= set(dictionary[field])

        for dictionary in dictionary_list:
            for value in values_set:
                if value in dictionary[field]:
                    dictionary[value] = 1
                else:
                    dictionary[value] = 0

        return dictionary_list

    @staticmethod
    def add_transpose_single_column(field, dictionary_list):
        """
        Takes a list of dictionaries and adds to every dictionary a new field
        for each value contained in the specified field among all the
        dictionaries in the field, leaving 1 for the values that are present in
        the dictionary and 0 for the values that are not. It can be seen as
        transposing the dictionary matrix.

        :param field: the field which is going to be transposed
        :param dictionary_list: a list of dictionaries
        :return: the modified list of dictionaries
        """

        values_set = set()
        for dictionary in dictionary_list:
            values_set.add(dictionary[field])

        for dictionary in dictionary_list:
            for value in values_set:
                if value in dictionary[field]:
                    dictionary[value] = 1
                else:
                    dictionary[value] = 0

        return dictionary_list

    @staticmethod
    def search_sentences(tips, noun):

        sentence_tokenizer = nltk.data.load(
            'tokenizers/punkt/english.pickle')
        stemmer = nltk.stem.SnowballStemmer('english')
        stemmed_noun = stemmer.stem(noun)

        founded_tips = []

        for tip in tips:
            sentences = sentence_tokenizer.tokenize(tip)
            for sentence in sentences:
                if stemmed_noun in sentence:
                    founded_tips.append(sentence)

        return founded_tips


    @staticmethod
    def split_train_test(records, split=0.8, shuffle_data=True, start=0.):
        """
        Splits the data in two disjunct datasets: train and test

        :param split: % of training set to be used (test set size = 100-percent)
        :type split: float
        :param shuffle_data: shuffle dataset?
        :type shuffle_data: bool

        :returns: a tuple <Data, Data>
        """
        if shuffle_data:
            np.random.shuffle(records)
        length = len(records)
        split_start = split + start

        if start == 0:
            train = records[:int(round(split*length))]
            test = records[int(round(split*length)):]
        elif split_start > 1:
            train = records[int(round(start*length)):] + records[:int(round((split_start-1)*length))]
            test = records[int(round((split_start-1)*length)):int(round(start*length))]
        else:
            train = records[int(round(start*length)):int(round(split_start*length))]
            test = records[int(round(split_start*length)):] + records[:int(round(start*length))]

        return train, test

    @staticmethod
    def load_csv_file(file_path, delimiter=','):

        records = []

        with open(file_path) as read_file:
            reader = csv.DictReader(read_file, delimiter=delimiter)  # read rows into a dictionary format
            for row in reader:
                dictionary = {}
                for (key, value) in row.items(): # go over each column name and value
                    dictionary[key] = value
                records.append(dictionary)

        return records

    @staticmethod
    def save_csv_file(
            file_path, records, headers, delimiter=',', write_headers=True):

        with open(file_path, 'wb') as write_file:
            writer = csv.DictWriter(write_file, headers, delimiter=delimiter)

            if write_headers:
                writer.writeheader()

            for record in records:
                writer.writerow(record)

    @staticmethod
    def json_to_csv(
            input_file, output_file, user_field, item_field, rating_field,
            shuffle=False, transform_ids=False, num_folds=None, delimiter=','):

        records = ETLUtils.load_json_file(input_file)
        fields = [user_field, item_field, rating_field]
        records = ETLUtils.select_fields(fields, records)
        if shuffle:
            random.shuffle(records)
        if transform_ids:
            records = ETLUtils.transform_ids(
                records, user_field, item_field, rating_field)

        if num_folds is None:
            ETLUtils.save_csv_file(
                output_file, records, fields, delimiter)
            return

        for fold in range(num_folds):
            split = 1 - (1/float(num_folds))
            start = float(fold) / num_folds
            train_records, test_records =\
                ETLUtils.split_train_test(records, split, False, start)

            ETLUtils.save_csv_file(
                output_file + "_train_" + str(fold), train_records, fields, delimiter)
            ETLUtils.save_csv_file(
                output_file + "_test_" + str(fold), test_records, fields, delimiter)

    @staticmethod
    def transform_ids(records, user_field, item_field, rating_field):
        user_id = 1
        item_id = 1

        user_map = {}
        item_map = {}

        new_records = []

        for record in records:

            old_user_id = record[user_field]
            old_item_id = record[item_field]

            if old_user_id not in user_map:
                user_map[old_user_id] = user_id
                user_id += 1

            if old_item_id not in item_map:
                item_map[old_item_id] = item_id
                item_id += 1

            new_record = {
                user_field:   user_map[old_user_id],
                item_field:   item_map[old_item_id],
                rating_field: record[rating_field]
            }

            new_records.append(new_record)

        return new_records


# headers = ['Algorithm',
#            # 'Multi-cluster',
#            # 'Similarity',
#            # 'Distance metric',
#            # 'Dataset',
#            # 'MAE	RMSE',
#            # 'Execution time',
#            # 'Cross validation',
#            'Machine']
# records = [
#     {'Algorithm': 'Clu_Overall', 'Machine': 'Mac'},
#     {'Algorithm': 'Clu_CF_Euc', 'Machine': 'Mac'},
#     {'Algorithm': 'Single_CF', 'Machine': 'PC'}
# ]
# ETLUtils.save_csv_file('/Users/fpena/tmp/test.csv', records, headers, delimiter='|')

my_input_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
my_input_file = my_input_folder + 'yelp_training_set_review_hotels.json'
my_fields = ['user_id', 'business_id', 'stars']
my_records = ETLUtils.load_json_file(my_input_file)
my_records = ETLUtils.select_fields(my_fields, my_records)
random.shuffle(my_records)

my_export_folder = '/Users/fpena/tmp/libfm-1.42.src/scripts/'
# my_export_file = my_export_folder + 'yelp_training_set_review_hotels_shuffled.csv'
my_export_file = my_export_folder + 'yelp.csv'
# ETLUtils.save_csv_file(my_export_file, my_records, my_fields, '\t', False)

# my_new_records = ETLUtils.transform_ids(my_records, 'user_id', 'business_id', 'stars')

# for my_new_record in my_new_records:
#     print(my_new_record)

# ETLUtils.json_to_csv(my_input_file, my_export_file, 'user_id', 'business_id', 'stars', True, True)
#
# print(my_records[0])
# print(my_records[2])
# print(my_records[5])
#
# data_frame = DataFrame(my_records)
# column = 'user_id'
# counts = data_frame.groupby(column).size()
# print(len(counts))

