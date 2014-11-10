import math
from pandas import DataFrame
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
        self.item_ids = extractor.get_groupby_list(self.reviews, 'offering_id')
        self.num_reviews = len(self.reviews)
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self.data_frame = DataFrame(self.reviews)
        self.users_count = self.data_frame.groupby('user_id').size()
        self.items_count = self.data_frame.groupby('offering_id').size()

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

    def calculate_sparsity_approx(self):
        """
        Returns the approximate percentage of missing ratings in the list of
        reviews of this ReviewsDatasetAnalyzer. This method is an approximation
        because it counts two reviews from the same user to the same item as
        two, when the correct count should be one. This method was created to
        calculate the sparsity in very big datasets where calculating the exact
        sparsity can be a very slow process.

        :return: the rate of approximate missing ratings
        (i.e. number of missing ratings / (number of reviews))
        :raise ValueError: in case an empty list is given
        """
        if not self.reviews:
            raise ValueError('Can not determine the sparsity for an empty list')

        user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        item_ids = extractor.get_groupby_list(self.reviews, 'offering_id')
        total_expected_reviews = float(len(user_ids) * len(item_ids))

        return 1 - float(len(self.reviews)) / total_expected_reviews

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

    def summarize_reviews_by_field(self, field):

        counts = self.data_frame.groupby(field).size()

        values = counts.values
        values_data_frame = DataFrame(values, columns=['frequency'])
        values_count = values_data_frame.groupby('frequency').size()

        return values_count

    @staticmethod
    def nCr(n, r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)


file_path = '/Users/fpena/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/filtered_reviews.json'
# file_path = '/Users/fpena/tmp/filtered_reviews_multi_non_sparse_shuffled.json'
my_reviews = ETLUtils.load_json_file(file_path)
# my_reviews = extractor.pre_process_reviews()
# my_reviews = movielens_extractor.get_ml_100K_dataset()
#
reviewsDatasetAnalyzer = ReviewsDatasetAnalyzer(my_reviews)
# my_counts = reviewsDatasetAnalyzer.summarize_reviews_by_field('user_id')
# file_name = '/Users/fpena/UCC/Thesis/projects/yelp/notebooks/test.ipynb'
# my_load_reviews_code =\
#     'file_path = \'/Users/fpena/tmp/filtered_reviews_multi_non_sparse_shuffled.json\'\n' +\
#     'reviews = ETLUtils.load_json_file(file_path)\n'
# my_load_reviews_code =\
#     'from tripadvisor.fourcity import movielens_extractor\n' +\
#     'reviews = movielens_extractor.get_ml_100K_dataset()'
# ReviewsDatasetAnalyzer.generate_report(my_reviews, 'Fourcity TripAdvisor', file_name, my_load_reviews_code)
# print(my_counts)
# print(reviewsDatasetAnalyzer.count_reviews_by_item())
# common_item_counts = reviewsDatasetAnalyzer.count_items_in_common()
# print(common_item_counts)
# print (time.strftime("%H:%M:%S"))
# print('Sparsity', reviewsDatasetAnalyzer.calculate_sparsity())
# print (time.strftime("%H:%M:%S"))
# print('Sparsity Approx', reviewsDatasetAnalyzer.calculate_sparsity_approx())
# print (time.strftime("%H:%M:%S"))
# print(reviewsDatasetAnalyzer.analyze_common_items_count(common_item_counts))
# print(reviewsDatasetAnalyzer.analyze_common_items_count(common_item_counts, True))

# print(reviewsDatasetAnalyzer.summarize_reviews_by_field('user_id'))
