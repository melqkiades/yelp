from DataPlotter import DataPlotter
import matplotlib.pyplot as plt


class YelpDataMain:
    def __init__(self):
        pass

    @staticmethod
    def plot_business_stars():
        data_folder = 'E:/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/'
        path = data_folder + 'yelp_academic_dataset_business.json'

        stars_plot = DataPlotter.plot_json_file(path, 'stars', 'bar',
                                                'Businesses\' ratings',
                                                'Rating',
                                                'Number of places', False)
        plt.show()

    @staticmethod
    def plot_business_reviews():
        data_folder = 'E:/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/'
        path = data_folder + 'yelp_academic_dataset_business.json'

        reviews_plot = DataPlotter.plot_json_file(path, 'review_count', 'line',
                                                  'Reviews per business',
                                                  'Review count',
                                                  'Frequency', True, True,
                                                  'log', 'log')
        plt.show()

    @staticmethod
    def plot_user_stars():
        data_folder = 'E:/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/'
        path = data_folder + 'yelp_academic_dataset_user.json'

        stars_plot = DataPlotter.plot_json_file(path, 'average_stars', 'line',
                                                'Reviews per user', 'Rating',
                                                'Average grade by users', False)
        plt.show()

    @staticmethod
    def plot_user_reviews():
        data_folder = 'E:/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/'
        path = data_folder + 'yelp_academic_dataset_user.json'

        reviews_plot = DataPlotter.plot_json_file(path, 'review_count', 'line',
                                                  'Reviews per user',
                                                  'Review count',
                                                  'Frequency', True, True,
                                                  'log', 'log')
        plt.show()

    @staticmethod
    def plot_review_stars():
        data_folder = 'E:/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/'
        path = data_folder + 'yelp_academic_dataset_review.json'

        stars_plot = DataPlotter.plot_json_file(path, 'stars', 'bar', 'Reviews',
                                                'Rating',
                                                'Frequency', False)
        plt.show()
