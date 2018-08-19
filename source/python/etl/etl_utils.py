import copy
import csv
import json
import os
import nltk

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
    def keep_fields(fields, dictionary_list):
        """
        Removes all but the specified fields from every dictionary in the
        dictionary list

        :rtype : void
        :param fields: a list of strings, which contains the fields that are
        going to be kept from every dictionary in the dictionary list
        :param dictionary_list: a list of dictionaries
        """
        unwanted_fields = set(dictionary_list[0]) - set(fields)
        for dictionary in dictionary_list:
            for unwanted_field in unwanted_fields:
                del dictionary[unwanted_field]

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
    def split_train_test(records, split=0.8, start=0.):
        """
        Splits the data in two disjunct datasets: train and test

        :param start: the position in which the split should start. This value
        must be in the range [0,1]
        :param records: the list to be split
        :param split: % of training set to be used (test set size = 100-percent)
        :type split: float

        :returns: a tuple <Data, Data>
        """
        length = len(records)
        split_start = split + start

        if start == 0:
            train = records[:int(round(split*length))]
            test = records[int(round(split*length)):]
        elif split_start > 1:
            train = records[int(round(start*length)):] +\
                records[:int(round((split_start-1)*length))]
            test = records[
                   int(round((split_start-1)*length)):int(round(start*length))]
        else:
            train = records[
                    int(round(start*length)):int(round(split_start*length))]
            test = records[int(round(split_start*length)):] +\
                records[:int(round(start*length))]

        return train, test

    @staticmethod
    def split_train_test_copy(records, split=0.8, start=0.):
        """
        Splits the data in two distinct datasets: train and test by making a
        deep copy of the records

        :param start: the position in which the split should start. This value
        must be in the range [0,1]
        :param records: the list to be split
        :param split: % of training set to be used (test set size = 100-percent)
        :type split: float

        :returns: a tuple <Data, Data>
        """
        records = copy.deepcopy(records)
        length = len(records)
        split_start = split + start

        if start == 0:
            train = records[:int(round(split*length))]
            test = records[int(round(split*length)):]
        elif split_start > 1:
            train = records[int(round(start*length)):] +\
                records[:int(round((split_start-1)*length))]
            test = records[
                   int(round((split_start-1)*length)):int(round(start*length))]
        else:
            train = records[
                    int(round(start*length)):int(round(split_start*length))]
            test = records[int(round(split_start*length)):] +\
                records[:int(round(start*length))]

        return train, test

    @staticmethod
    def load_csv_file(file_path, delimiter=',', fieldnames=None):

        records = []

        with open(file_path) as read_file:
            # read rows into a dictionary format
            reader = csv.DictReader(
                read_file, delimiter=delimiter, fieldnames=fieldnames)
            for row in reader:
                dictionary = {}
                # go over each column name and value
                for (key, value) in row.items():
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
    def write_row_to_csv(file_name, row):
        """
        Writes a new line to the end of the specified CSV file. The new line
        will be composed of the values of the row dictionary sorted by their
        key values in alphabetical order. If file_name does not exists, then it
        will be created automatically. The headers will be the keys of the row
        dictionary sorted in alphabetical order

        :param file_name: the name of the CSV file
        :param row: a dictionary
        """
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                w = csv.DictWriter(f, sorted(row.keys()))
                w.writeheader()
                w.writerow(row)
        else:
            with open(file_name, 'a') as f:
                w = csv.DictWriter(f, sorted(row.keys()))
                w.writerow(row)

    @staticmethod
    def write_row_to_json(file_name, row):
        """
        Writes a new line to the end of the specified JSON file. If file_name
        does not exists, then it will be created automatically.

        :param file_name: the name of the JSON file
        :param row: a dictionary
        """
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                json.dump(row, f)
                f.write('\n')
        else:
            with open(file_name, 'a') as f:
                json.dump(row, f)
                f.write('\n')

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

    @staticmethod
    def count_frequency(records, field):
        """
        Counts the times that every value of field appears in the list of
        records. It's the equivalent of a COUNT DISTINCT clause in SQL

        :type records: list[dict]
        :param records: a list of dictionaries
        :type field: str
        :param field: the field which contains the values to count

        :rtype: dict
        :return a dictionary where the keys are the distinct values that appear
         in the given field, and the values is the number of times they appear
         throughout all the list of records
        """

        frequency_map = {}

        for record in records:
            key = record[field]
            if key not in frequency_map:
                frequency_map[key] = 0
            frequency_map[key] += 1

        return frequency_map
