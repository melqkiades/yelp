import json

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
    def drop_fields(fields, dictionary_list):
        """
        Removes the specified fields from every dictionary in the list records

        :rtype : void
        :param fields: a list of strings, which contains the fields that are
        going to be removed from every dictionary in the list records
        :param dictionary_list: a list of dictionaries
        """
        for record in dictionary_list:
            for field in fields:
                del (record[field])

    @staticmethod
    def filter_records(records, field, values):
        filtered_records = [record for record in records if
                            record[field] in values]
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