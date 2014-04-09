import numpy
from etl import ETLUtils

__author__ = 'franpena'


class BusinessETL:
    def __init__(self):
        pass

    @staticmethod
    def create_category_matrix(file_path):
        """
        Creates a matrix with all the categories for businesses that are
        contained in the Yelp Phoenix Business data set. Each column of the
        matrix represents a category, and each row a business. This is a binary
        matrix that contains a 1 at the position i,j if the business i contains
        the category j, and a 0 otherwise.

        :rtype : numpy array matrix
        :param file_path: the path for the file that contains the businesses
        data
        :return: a numpy array binary matrix
        """
        records = ETLUtils.load_json_file(file_path)

        # Now we obtain the categories for all the businesses
        records = ETLUtils.add_transpose_list_column('categories', records)
        BusinessETL.drop_unwanted_fields(records)
        matrix = numpy.array(
            [numpy.array(record.values()) for record in records])

        return matrix

    @staticmethod
    def create_category_sets(file_path):
        """
        Creates an array of arrays in which each sub-array contains the
        categories of each business in the Yelp Phoenix Business data set

        :rtype : numpy array matrix
        :param file_path: the path for the file that contains the businesses
        data
        :return: a numpy array of numpy arrays with the categories that each
        business has, for example [['Restaurant', 'Mexican', 'Bar'],
        ['Bar', 'Disco']]
        """
        records = ETLUtils.load_json_file(file_path)
        sets = numpy.array(
            [set(record['categories']) for record in records])

        return sets


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
            'latitude',
            'longitude',
            'hours',
            'name',
            'neighborhoods',
            'open',
            'review_count',
            'stars',
            'state',
            'type'
        ]

        ETLUtils.drop_fields(unwanted_fields, dictionary_list)


data_folder = '../../../../../../datasets/yelp_phoenix_academic_dataset/'
business_file_path = data_folder + 'yelp_academic_dataset_business.json'
my_records = BusinessETL.create_category_sets(business_file_path)
