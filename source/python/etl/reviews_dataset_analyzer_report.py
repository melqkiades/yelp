from etl import ETLUtils
from etl.reviews_dataset_analyzer import ReviewsDatasetAnalyzer
from tripadvisor.fourcity import movielens_extractor
from IPython.nbformat import current as nbf

__author__ = 'fpena'


class ReviewsDatasetAnalyzerReport:

    @staticmethod
    def generate_report(reviews, dataset_name, file_name, load_reviews_code):

        nb = nbf.new_notebook()
        title = '# ' + dataset_name + ' Dataset Analysis'
        title_cell = nbf.new_text_cell(u'markdown', title)

        rda = ReviewsDatasetAnalyzer(reviews)
        num_reviews = len(rda.reviews)
        num_users = len(rda.user_ids)
        num_items = len(rda.item_ids)
        user_avg_reviews = float(num_reviews) / num_users
        item_avg_reviews = float(num_reviews) / num_items
        sparsity = rda.calculate_sparsity_approx()

        fact_sheet_text =\
            '## Fact Sheet\n' +\
            'The ' + dataset_name + ' contains:\n' +\
            '* ' + str(num_reviews) + ' reviews\n' +\
            '* Made by ' + str(num_users) + ' users\n' +\
            '* About ' + str(num_items) + ' items\n' +\
            '* It has an approximated sparsity of ' + str(sparsity) + '\n' +\
            '\nNow we are going to analyze the number of reviews per user and ' \
            'per item'
        fact_sheet_cell = nbf.new_text_cell(u'markdown', fact_sheet_text)

        reviews_analysis_code =\
            'import sys\n' +\
            'sys.path.append(\'/Users/fpena/UCC/Thesis/projects/yelp/source/python\')\n' +\
            'from etl import ETLUtils\n\n' +\
            'from etl.reviews_dataset_analyzer import ReviewsDatasetAnalyzer\n' +\
            '\n# Load reviews\n' + load_reviews_code + '\n' +\
            'rda = ReviewsDatasetAnalyzer(reviews)\n'
        reviews_analysis_cell = nbf.new_code_cell(reviews_analysis_code)

        user_analysis_text =\
            '## Users Reviews Analysis\n' +\
            '* The average number of reviews per user is ' + str(user_avg_reviews) + '\n' +\
            '* The minimum number of reviews a user has is ' + str(min(rda.users_count)) + '\n' +\
            '* The maximum number of reviews a user has is ' + str(max(rda.users_count))
        user_analysis_cell = nbf.new_text_cell(u'markdown', user_analysis_text)

        counts_per_user_code =\
            '# Number of reviews per user\n' +\
            'users_summary = rda.summarize_reviews_by_field(\'user_id\')\n' +\
            'print(\'Average number of reviews per user\', float(rda.num_reviews)/rda.num_users)\n' +\
            'users_summary.plot(kind=\'line\', rot=0)'
        counts_per_user_cell = nbf.new_code_cell(counts_per_user_code)

        item_analysis_text =\
            '## Items Reviews Analysis\n' +\
            '* The average number of reviews per item is ' + str(item_avg_reviews) + '\n' +\
            '* The minimum number of reviews an item has is ' + str(min(rda.items_count)) + '\n' +\
            '* The maximum number of reviews an item has is ' + str(max(rda.items_count))
        item_analysis_cell = nbf.new_text_cell(u'markdown', item_analysis_text)

        counts_per_item_code =\
            '# Number of reviews per item\n' +\
            'items_summary = rda.summarize_reviews_by_field(\'offering_id\')\n' +\
            'print(\'Average number of reviews per item\', float(rda.num_reviews)/rda.num_items)\n' +\
            'items_summary.plot(kind=\'line\', rot=0)'
        counts_per_item_cell = nbf.new_code_cell(counts_per_item_code)

        common_items_text =\
            '## Number of items 2 users have in common\n' +\
            'In this section we are going to count the number of items two ' \
            'users have in common'
        common_items_text_cell = nbf.new_text_cell(u'markdown', common_items_text)

        common_items_code =\
            '# Number of items 2 users have in common\n' +\
            'common_item_counts = rda.count_items_in_common()\n' +\
            'plt.plot(common_item_counts.keys(), common_item_counts.values())\n'
        common_items_code_cell = nbf.new_code_cell(common_items_code)

        common_items_box_code =\
            'from pylab import boxplot\n' +\
            'my_data = [key for key, value in common_item_counts.iteritems() for i in xrange(value)]\n' +\
            'boxplot(my_data)'
        common_items_box_cell = nbf.new_code_cell(common_items_box_code)

        cells = []
        cells.append(title_cell)
        cells.append(fact_sheet_cell)
        cells.append(reviews_analysis_cell)
        cells.append(user_analysis_cell)
        cells.append(counts_per_user_cell)
        cells.append(item_analysis_cell)
        cells.append(counts_per_item_cell)
        cells.append(common_items_text_cell)
        cells.append(common_items_code_cell)
        cells.append(common_items_box_cell)
        nb['worksheets'].append(nbf.new_worksheet(cells=cells))

        with open(file_name, 'w') as f:
            nbf.write(nb, f, 'ipynb')
        # counts.plot(kind='bar', rot=0)

    @staticmethod
    def generate_report_ml100k():
        reviews = movielens_extractor.get_ml_100K_dataset()
        file_name = '/Users/fpena/UCC/Thesis/projects/yelp/notebooks/test_ml100k.ipynb'
        load_reviews_code =\
            'from tripadvisor.fourcity import movielens_extractor\n' +\
            'reviews = movielens_extractor.get_ml_100K_dataset()'
        dataset_name = 'Sorina Likes the New Asian Guy\'s Butt 100k'
        ReviewsDatasetAnalyzerReport.generate_report(
            reviews, dataset_name, file_name, load_reviews_code)

    @staticmethod
    def generate_report_fourcity_filtered():
        file_path = '/Users/fpena/tmp/filtered_reviews_multi_non_sparse_shuffled.json'
        file_name = '/Users/fpena/UCC/Thesis/projects/yelp/notebooks/test_fourcity.ipynb'
        reviews = ETLUtils.load_json_file(file_path)
        load_reviews_code =\
            'file_path = \'/Users/fpena/tmp/filtered_reviews_multi_non_sparse_shuffled.json\'\n' +\
            'reviews = ETLUtils.load_json_file(file_path)\n'
        ReviewsDatasetAnalyzerReport.generate_report(reviews, 'Fourcity TripAdvisor', file_name, load_reviews_code)

    @staticmethod
    def generate_report_ruihai():
        file_path = '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/cleaned_reviews.json'
        file_name = '/Users/fpena/UCC/Thesis/projects/yelp/notebooks/test_ruihai.ipynb'
        reviews = ETLUtils.load_json_file(file_path)
        load_reviews_code =\
            'file_path = \'/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/cleaned_reviews.json\'\n' +\
            'reviews = ETLUtils.load_json_file(file_path)\n'
        ReviewsDatasetAnalyzerReport.generate_report(reviews, 'Ruihai TripAdvisor', file_name, load_reviews_code)

ReviewsDatasetAnalyzerReport.generate_report_fourcity_filtered()
ReviewsDatasetAnalyzerReport.generate_report_ml100k()
ReviewsDatasetAnalyzerReport.generate_report_ruihai()