import random
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
        self.var_lambda = 0.001
        self.momemtum = 0.8
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
        batch_size = int(np.ceil(float(len(self.train) / self.num_batches)))
        self.alpha = self.epsilon / batch_size

        self.logger.debug('epsilon: %f', self.epsilon)
        self.logger.debug('alpha: %f', self.alpha)

        # latent variables
        self.user_features = 0.1 * np.random.rand(self.num_users, self.num_features)
        self.item_features = 0.1 * np.random.rand(self.num_items, self.num_features)

        converge = 1e-4
        last_rmse = None

        for epoch in xrange(self.num_epochs):

            for batch in xrange(self.num_batches):
                data = self.train[batch * batch_size: (batch + 1) * batch_size]

                # print "batch_size", batch_size
                # print "train", train.shape
                # print "data", data.shape

                # compute gradient
                user_features = self.user_features[data[:, 0], :]
                item_features = self.item_features[data[:, 1], :]
                # print "u_feature", user_features.shape
                # print "i_feature", item_features.shape

                ratings = data[:, 2] - self.mean_rating
                predictions = np.sum(user_features * item_features, 1)
                error_list = predictions - ratings

                # print "predictions", predictions.shape
                # print "errs", error_list.shape

                error_matrix = np.tile(error_list, (self.num_features, 1)).T
                # print "err_matrix", error_matrix.shape

                user_gradients =\
                    user_features * error_matrix + self.var_lambda * item_features
                item_gradients =\
                    item_features * error_matrix + self.var_lambda * user_features

                user_feature_gradients =\
                    np.zeros((self.num_users, self.num_features))
                item_feature_gradients =\
                    np.zeros((self.num_items, self.num_features))

                for i in xrange(data.shape[0]):
                    user = data[i, 0]
                    item = data[i, 1]
                    user_feature_gradients[user, :] += user_gradients[i, :]
                    item_feature_gradients[item, :] += item_gradients[i, :]

                # Update movie and user features

                self.user_features = self.user_features - \
                    self.alpha * user_feature_gradients
                self.item_features = self.item_features - \
                    self.alpha * item_feature_gradients

                # print('epsilon', self.epsilon)
                # print('batch size', batch_size)
                # print('user_feature_gradients', user_feature_gradients)
                # print('gradient', ((self.epsilon / batch_size) * user_feature_gradients))
                # print('user features')
                # print(self.user_features[0])
                # print(self.user_features[10])
                # print(self.user_features[100])

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

        # print(predictions[0])
        # print(predictions[10])
        # print(predictions[100])

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
    np.random.shuffle(ratings)

    # split data to training & validation
    train_pct = 0.9
    train_size = int(train_pct * len(ratings))
    train = ratings[:train_size]
    validation = ratings[train_size:]

    # params
    num_feature = 10
    bmf_model = ProbabilisticMatrixFactorization()

    start_time = time.clock()
    bmf_model.load(train, validation)
    end_time = time.clock()
    print "time spent = %.3f" % (end_time - start_time)

    return bmf_model

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')
example()


# random.seed(0)
#
# a = [0, 1, 2, 3, 4, 5]
# random.shuffle(a)
# print(a)
# np.random.seed(0)
# print (np.random.permutation(range(6)))
#
# print(np.random.random())
# print(np.random.random())
# print(np.random.random())
# print(np.random.random())
