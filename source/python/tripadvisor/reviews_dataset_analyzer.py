import math
from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


class ReviewsDatasetAnalyzer:
    """
    This class is used to analyze the dataset that contains a list of reviews.
    Some statistics can be obtained thanks to this class, such as the sparsity
    of the dataset and the number of users that have rated the same item. This
    statistics will show how complete is the dataset.
    """

    def __init__(self, reviews):

        if not reviews:
            raise ValueError('Can not analyze an empty list')

        self.reviews = reviews
        self.user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        self.num_users = len(self.user_ids)

    def calculate_sparsity(self):
        """
        Returns the percentage of missing ratings in the list of reviews of this
        ReviewsDatasetAnalyzer

        :return: the rate of missing ratings
        (i.e. number of missing ratings / (number of items * number of users))
        :raise ValueError: in case an empty list is given
        """
        if not self.reviews:
            raise ValueError('Can not determine the sparsity for an empty list')

        user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        item_ids = extractor.get_groupby_list(self.reviews, 'offering_id')

        non_missing_reviews = 0.
        total_expected_reviews = len(user_ids) * len(item_ids)

        for user in user_ids:
            user_reviews = ETLUtils.filter_records(self.reviews, 'user_id', [user])
            user_items = extractor.get_groupby_list(user_reviews, 'offering_id')

            non_missing_reviews += len(set(item_ids).intersection(set(user_items)))

        return 1 - non_missing_reviews / total_expected_reviews

    def count_items_in_common(self):
        """
        Counts the number of items each user has in common with every other user
        and stores the results in a dictionary.

        :return: a dictionary with the count of the number of times users have
        a certain number of items in common. For example, the dictionary
        {0:4, 1:10, 2:6, 3:3, 4:1} means that there a 4 users who have 0 items
        in common with the rest of users, there are 10 users who have rated 1
        item in common with the rest of users, 6 users who have rated 6 items in
        common with the rest of the users, and so on.
        """
        common_item_counts = {}
        user_dictionary = extractor.initialize_users(self.reviews, False)

        for i in range(self.num_users):
            for j in range(i+1, self.num_users):

                user1 = self.user_ids[i]
                user2 = self.user_ids[j]

                num_common_items = len(extractor.get_common_items(
                    user_dictionary, user1, user2))

                if num_common_items in common_item_counts:
                    common_item_counts[num_common_items] += 1
                else:
                    common_item_counts[num_common_items] = 1

        return common_item_counts

    def analyze_common_items_count(self, common_item_counts, cumulative=False):
        """
        Converts the dictionary returned by
        :func:`~reviews_dataset_analyzer.ReviewsDatasetAnalyzer.count_items_in_common`
        into a dictionary with the same keys but instead of counts the values
        have percentages

        :param common_item_counts: a dictionary with the count of the number of
        items the users have in common
        :param cumulative: indicates whether the percentage is cumulative or not
        :return: a dictionary with the same keys but instead of counts the
        values have percentages
        """
        num_combinations = ReviewsDatasetAnalyzer.nCr(self.num_users, 2)
        common_item_percentages = {}

        cumulative_percentage = 0.

        for key in common_item_counts.keys():
            percentage = float(common_item_counts[key]) / num_combinations

            if cumulative:
                cumulative_percentage += percentage
                common_item_percentages[key] = cumulative_percentage
            else:
                common_item_percentages[key] = percentage

        return common_item_percentages

    @staticmethod
    def nCr(n, r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)


# for i in range(10):
#     for j in range(i+1, 10):
#         print(i, j)
