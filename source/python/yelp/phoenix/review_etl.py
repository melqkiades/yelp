import json
from operator import itemgetter
import time

from etl import ETLUtils
from yelp.phoenix.business_etl import BusinessETL

__author__ = 'franpena'


class ReviewETL:
    def __init__(self):
        pass

    @staticmethod
    def load_file(file_path):
        """
        Loads the Yelp Phoenix Academic Data Set file for business data, and
        transforms it so it can be analyzed

        :type file_path: string
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


    # @staticmethod
    # def filter_reviews_by_business(reviews, business_ids, field=None):
    #     # if not field:
    #     #     return [review for review in reviews
    #     #             if review['business_id'] in business_ids]
    #     # return [review[field] for review in reviews
    #     #         if review['business_id'] in business_ids]
    #
    #     reviews.sort(key=itemgetter('business_id'), reverse=False)
    #     business_ids.sort(reverse=False)
    #
    #     num_reviews = len(business_ids)
    #     filtered_reviews = []
    #     flag = False
    #     business_index = 0
    #
    #     for review in reviews:
    #         if business_index >= num_reviews:
    #             break
    #         if review['business_id'] == business_ids[business_index]:
    #             filtered_reviews.append(review)
    #             flag = True
    #         elif flag:
    #             flag = False
    #             business_index += 1
    #
    #     return filtered_reviews

    @staticmethod
    def filter_reviews_by_business_slow(reviews, business_ids, field=None):
        # if not field:
        #     return [review for review in reviews
        #             if review['business_id'] in business_ids]
        # return [review[field] for review in reviews
        #         if review['business_id'] in business_ids]
        filtered_reviews = []

        for review in reviews:
            if review['business_id'] in business_ids:
                filtered_reviews.append(review)

        return filtered_reviews

    @staticmethod
    def sort_records(records, field, reverse=False):
        return sorted(records, key=itemgetter(field), reverse=reverse)



start = time.time()

review_etl = ReviewETL()
my_business_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_business.json"
my_reviews_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review.json"
my_business_ids = BusinessETL.get_business_ids(my_business_file, 'Hotels')
my_reviews = ETLUtils.load_json_file(my_reviews_file)
# print(len(ReviewETL.filter_reviews_by_business(my_reviews, my_business_ids, 'text')))
my_restaurant_reviews = ReviewETL.filter_reviews_by_business_slow(my_reviews, my_business_ids)
my_restaurants_file = "/Users/fpena/tmp/yelp_training_set/yelp_training_set_review_hotels.json"
ETLUtils.save_json_file(my_restaurants_file, my_restaurant_reviews)
# my_sorted_reviews = ReviewETL.sort_records(my_reviews, 'business_id')
# print(len(my_sorted_reviews))


# main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
