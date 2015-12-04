import cPickle as pickle
import time

import numpy
from sklearn.linear_model import LogisticRegression

from etl import ETLUtils
from topicmodeling.context import review_metrics_extractor
from topicmodeling.context.review import Review

__author__ = 'fpena'


class ReviewsClassifier:

    def __init__(self):
        self.num_features = 2
        self.classifier = LogisticRegression(C=100)
        self.min_values = None
        self.max_values = None

    def train(self, records, reviews=None):

        if reviews is None:
            reviews = []
            for record in records:
                reviews.append(Review(record['text']))

        if len(records) != len(reviews):
            msg = 'The size of the records and reviews arrays must be the same'
            raise ValueError(msg)

        metrics = numpy.zeros((len(reviews), self.num_features))

        for index in range(len(reviews)):
            metrics[index] =\
                review_metrics_extractor.get_review_metrics(reviews[index])

        self.min_values = metrics.min(axis=0)
        self.max_values = metrics.max(axis=0)
        review_metrics_extractor.normalize_matrix_by_columns(
            metrics, self.min_values, self.max_values)

        labels = numpy.array([record['specific'] == 'yes' for record in records])
        self.classifier.fit(metrics, labels)

    def predict(self, reviews):
        metrics = numpy.zeros((len(reviews), self.num_features))
        for index in range(len(reviews)):
            metrics[index] =\
                review_metrics_extractor.get_review_metrics(reviews[index])

        review_metrics_extractor.normalize_matrix_by_columns(
            metrics, self.min_values, self.max_values)
        return self.classifier.predict(metrics)

    def label_json_reviews(self, input_file, output_file, reviews=None):

        records = ETLUtils.load_json_file(input_file)

        if reviews is None:
            reviews = []
            for record in records:
                reviews.append(Review(record['text']))

        if len(records) != len(reviews):
            msg = 'The size of the records and reviews arrays must be the same'
            raise ValueError(msg)
        predicted_classes = self.predict(reviews)

        for record, predicted_class in zip(records, predicted_classes):
            if predicted_class:
                label = 'specific'
            else:
                label = 'generic'

            record['predicted_class'] = label

        ETLUtils.save_json_file(output_file, records)


def main():
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    my_training_records_file = my_folder + 'classified_restaurant_reviews.json'
    my_training_reviews_file = my_folder + 'classified_restaurant_reviews.pkl'
    my_training_records = ETLUtils.load_json_file(my_training_records_file)

    with open(my_training_reviews_file, 'rb') as read_file:
        my_training_reviews = pickle.load(read_file)

    classifier = ReviewsClassifier()
    classifier.train(my_training_records, my_training_reviews)

    my_input_records_file = my_folder + 'yelp_training_set_review_restaurants_shuffled.json'
    my_input_reviews_file = my_folder + 'reviews_restaurant_shuffled.pkl'
    my_output_records_file = my_folder + 'yelp_training_set_review_restaurants_shuffled_tagged.json'

    with open(my_input_reviews_file, 'rb') as read_file:
        my_input_reviews = pickle.load(read_file)

    classifier.label_json_reviews(
        my_input_records_file, my_output_records_file, my_input_reviews)


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
