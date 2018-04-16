import numpy

from topicmodeling.context import review_metrics_extractor
from utils.constants import Constants

__author__ = 'fpena'


class ReviewsClassifier:

    def __init__(self, classifier):
        self.num_features = None
        self.min_values = None
        self.max_values = None
        self.classifier = classifier

    def transform(self, records):
        """
        Transforms the reviews into a numpy matrix so that they can be easily
        processed by the functions available in scikit-learn

        :type records: list[dict]
        :param records: a list of dictionaries with the reviews
        :return: a matrix with the independent variables (X) and a vector with
        the dependent variables (y)
        """

        self.num_features = \
            len(review_metrics_extractor.get_review_metrics(records[0]))
        metrics = numpy.zeros((len(records), self.num_features))

        for index in range(len(records)):
            metrics[index] = \
                review_metrics_extractor.get_review_metrics(records[index])

        self.min_values = metrics.min(axis=0)
        self.max_values = metrics.max(axis=0)
        review_metrics_extractor.normalize_matrix_by_columns(
            metrics, self.min_values, self.max_values)

        labels = \
            numpy.array([record['specific'] == 'yes' for record in records])

        return metrics, labels

    def train(self, records):

        metrics, labels = self.transform(records)

        # if self.sampler is not None:
        #     metrics, labels = self.sampler.fit_sample(metrics, labels)

        self.classifier.fit(metrics, labels)

    def predict(self, records):
        metrics = numpy.zeros((len(records), self.num_features))
        for index in range(len(records)):
            metrics[index] =\
                review_metrics_extractor.get_review_metrics(records[index])

        review_metrics_extractor.normalize_matrix_by_columns(
            metrics, self.min_values, self.max_values)
        return self.classifier.predict(metrics)

    def label_json_reviews(self, records):

        predicted_classes = self.predict(records)

        for record, predicted_class in zip(records, predicted_classes):
            if predicted_class:
                label = Constants.SPECIFIC
            else:
                label = Constants.GENERIC

            record[Constants.PREDICTED_CLASS_FIELD] = label

    def score(self, records):
        metrics = numpy.zeros((len(records), self.num_features))
        for index in range(len(records)):
            metrics[index] = \
                review_metrics_extractor.get_review_metrics(records[index])

        review_metrics_extractor.normalize_matrix_by_columns(
            metrics, self.min_values, self.max_values)

        labels = \
            numpy.array([record['specific'] == 'yes' for record in records])

        return self.classifier.score(metrics, labels)
