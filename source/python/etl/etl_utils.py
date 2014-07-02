import json
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
    def drop_fields(fields, dictionary_list):
        """
        Removes the specified fields from every dictionary in the dictionary
        list

        :rtype : void
        :param fields: a list of strings, which contains the fields that are
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