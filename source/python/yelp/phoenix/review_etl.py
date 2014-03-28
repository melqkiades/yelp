import json

from etl import ETLUtils


__author__ = 'franpena'


class ReviewETL:
    def __init__(self):
        pass

    @staticmethod
    def load_file(file_path):
        """
        Loads the Yelp Phoenix Academic Data Set file for business data, and
        transforms it so it can be analyzed

        :type file_path: list of dictionaries
        :param file_path: the path for the file that contains the businesses
        data
        :return: a list of dictionaries with the preprocessed data
        """
        records = [json.loads(line) for line in open(file_path)]

        # Now we obtain the categories for all the businesses
        records = ReviewETL.add_transpose_list_column('categories', records)
        records = ReviewETL.add_transpose_single_column('city', records)
        ReviewETL.drop_unwanted_fields(records)

        return records

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
    def drop_unwanted_fields(dictionary_list):
        """
        Drops fields that are not useful for data analysis in the business
        data set

        :rtype : void
        :param dictionary_list: the list of dictionaries containing the data
        """
        unwanted_fields = [
            'attributes',
            'business_id',
            'categories',
            'city',
            'full_address',
            'hours',
            'name',
            'neighborhoods',
            'open',
            'state',
            'type'
        ]

        ETLUtils.drop_fields(unwanted_fields, dictionary_list)




review = ReviewETL()