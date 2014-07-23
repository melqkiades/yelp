from etl import ETLUtils
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity.user import User

__author__ = 'fpena'


class ReviewsHolder:

    def __init__(self, reviews):
        self.reviews = reviews
        self.user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        self.user_dictionary = self.initialize_users(self.reviews)

    @staticmethod
    def initialize_users(reviews):
        """
        Builds a dictionary containing all the users in the reviews. Each user
        contains information about its average overall rating, the list of reviews
        that user has made, and the cluster the user belongs to

        :param reviews: the list of reviews
        :return: a dictionary with the users initialized, the keys of the
        dictionaries are the users' ID
        """
        user_ids = extractor.get_groupby_list(reviews, 'user_id')
        user_dictionary = {}

        for user_id in user_ids:
            user = User(user_id)
            user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
            user.average_overall_rating =\
                extractor.get_user_average_overall_rating(
                    user_reviews, user_id, apply_filter=False)
            user_dictionary[user_id] = user
            user.item_ratings = extractor.get_user_item_ratings(user_reviews)

        return user_dictionary

    def get_rating(self, user, item):
        if item in self.user_dictionary[user].item_ratings:
            return self.user_dictionary[user].item_ratings[item]
        return None

    def get_user_average_rating(self, user_id):
        return self.user_dictionary[user_id].average_overall_rating

    def get_common_items(self, user1, user2):
        items_user1 = set(self.user_dictionary[user1].item_ratings.keys())
        items_user2 = set(self.user_dictionary[user2].item_ratings.keys())

        common_items = items_user1.intersection(items_user2)

        return common_items

    def extract_user_ratings(self, user, items):

        ratings = []

        for item in items:
            ratings.append(self.get_rating(user, item))

        return ratings