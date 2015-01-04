import time
import logging
from tripadvisor.fourcity import movielens_extractor

__author__ = 'fpena'

import numpy as np


# Sample hyperparameters

class ProbabilisticMatrixFactorization():

    def __init__(self):

        self.logger = logging.getLogger(type(self).__name__)

        # We define the variables necessary to build the matrix
        self.epsilon = 50.0
        self.var_lambda = 0.01
        self.momentum = 0.8
        self.alpha = 0.0001

        self.max_rating = 5
        self.min_rating = 1

        # self.epoch = 1
        self.num_epochs = 50

        # Load the data and from it obtain the training and validation sets

        self.train_vec = None
        self.probe_vec = None

        self.mean_rating = None  # Calculate the mean rating from the training set

        self.train = None
        self.validation = None
        self.num_batches = 9
        self.batch_size = None
        self.num_items = 3952  # Can be calculated
        self.num_users = 6040  # Can be calculated
        self.num_features = 10
        self.user_features = None
        self.item_features = None
        self.train_errors = []
        self.validation_errors = []


    def load(self, train, validation):

        # self.train =\
        #     movielens_extractor.reviews_to_numpy_matrix(train_reviews)
        self.train = train
        self.validation = validation
        self.num_users = movielens_extractor.get_num_users(self.train)
        self.num_items = movielens_extractor.get_num_items(self.train)
        self.mean_rating = movielens_extractor.get_mean_rating(self.train)
        self.batch_size = int(np.ceil(float(len(self.train) / self.num_batches)))
        self.alpha = self.epsilon / self.batch_size

        self.logger.debug('epsilon: %f', self.epsilon)
        self.logger.debug('alpha: %f', self.alpha)

        # latent variables
        self.user_features = 0.1 * np.random.rand(self.num_users, self.num_features)
        self.item_features = 0.1 * np.random.rand(self.num_items, self.num_features)
        user_features_inc = np.zeros((self.num_users, self.num_features))
        item_features_inc = np.zeros((self.num_items, self.num_features))

        converge = 1e-4
        last_rmse = None

        for epoch in xrange(self.num_epochs):

            # In each cycle the training vector is shuffled
            np.random.shuffle(self.train)

            for batch in xrange(self.num_batches):
                data = self.train[batch * self.batch_size: (batch + 1) * self.batch_size]
                users = data[:, 0]
                items = data[:, 1]
                ratings = data[:, 2]

                 # Default prediction is the mean rating
                ratings = ratings - self.mean_rating

                # compute predictions
                predicted_ratings = np.sum(self.user_features[users, :] * self.item_features[items, :], 1)

                # compute gradients
                error_list = predicted_ratings - ratings
                error_matrix = np.tile(error_list, (self.num_features, 1)).T

                item_gradients =\
                    error_matrix * self.user_features[users, :] + self.var_lambda * self.item_features[items, :]
                user_gradients =\
                    error_matrix * self.item_features[items, :] + self.var_lambda * self.user_features[users, :]

                user_feature_gradients =\
                    np.zeros((self.num_users, self.num_features))
                item_feature_gradients =\
                    np.zeros((self.num_items, self.num_features))
                # In the above line the gradient is calculated for every rating,
                #  but it has to be grouped (by summing it) for each user and
                # item features

                for i in xrange(self.batch_size):
                    user_feature_gradients[users[i], :] += user_gradients[i, :]
                    item_feature_gradients[items[i], :] += item_gradients[i, :]

                # Update item and user features
                # The update is done using the momentum technique of gradient descent
                user_features_inc = self.momentum * user_features_inc +\
                    self.alpha * user_feature_gradients
                # self.user_features = self.user_features - \
                #     self.alpha * user_feature_gradients
                self.user_features = self.user_features - user_features_inc

                item_features_inc = self.momentum * item_features_inc +\
                    self.alpha * item_feature_gradients
                # self.item_features = self.item_features - \
                #     self.alpha * item_feature_gradients
                self.item_features = self.item_features - item_features_inc

            # compute RMSE
            # train errors

            train_preds = self.predict(self.train)
            train_rmse = RMSE(train_preds, np.float16(self.train[:, 2]))

            # validation errors
            validation_preds = self.predict(self.validation)
            validation_rmse = RMSE(
                validation_preds, np.float16(self.validation[:, 2]))
            self.train_errors.append(train_rmse)
            self.validation_errors.append(validation_rmse)
            print "iterations: %3d, train RMSE: %.6f, validation RMSE: %.6f " % \
                (epoch + 1, train_rmse, validation_rmse)

            # stop if converge
            if last_rmse:
                if abs(train_rmse - last_rmse) < converge:
                    # break
                    pass
            last_rmse = train_rmse

    def predict(self, data):
        user_features = self.user_features[data[:, 0], :]
        item_features = self.item_features[data[:, 1], :]
        predictions =\
            np.sum(user_features * item_features, 1) + self.mean_rating

        if self.max_rating:
            predictions[predictions > self.max_rating] = self.max_rating

        if self.min_rating:
            predictions[predictions < self.min_rating] = self.min_rating

        return predictions


def RMSE(estimation, truth):
    """Root Mean Square Error"""

    num_sample = len(estimation)

    # sum square error
    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1.0))


def example():
    """simple test and performance measure
    """
    reviews = movielens_extractor.get_ml_1m_dataset()
    ratings = movielens_extractor.reviews_to_numpy_matrix(reviews)
    # suffle_data
    np.random.seed(0)
    np.random.shuffle(ratings)

    # split data to training & validation
    train_pct = 0.9
    train_size = int(train_pct * len(ratings))
    train = ratings[:train_size]
    validation = ratings[train_size:]

    # params
    bmf_model = ProbabilisticMatrixFactorization()

    start_time = time.clock()
    bmf_model.load(train, validation)
    end_time = time.clock()
    print "time spent = %.3f" % (end_time - start_time)

    return bmf_model

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')
example()

