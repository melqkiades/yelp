import numpy.random as rand
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
import time
from tripadvisor.fourcity import movielens_extractor

__author__ = 'fpena'

import numpy as np


class BayesianMatrixFactorization:

    def __init__(self):

        self.num_features = 10

        self.num_users = None
        self.num_items = None
        self.train = None
        self.validation = None

        self.mean_rating = None

        self.max_rating = 5
        self.min_rating = 1

        # Hyper Parameter
        self.beta = 2.0
        # Inv-Whishart (User features)
        self.WI_user = np.eye(self.num_features, dtype='float16')
        self.beta_user = 2.0
        self.df_user = self.num_features
        self.mu0_user = np.zeros((self.num_features, 1), dtype='float16')

        # Inv-Whishart (item features)
        self.WI_item = np.eye(self.num_features, dtype='float16')
        self.beta_item = 2.0
        self.df_item = self.num_features
        self.mu0_item = np.zeros((self.num_features, 1), dtype='float16')

        # Latent Variables
        self.mu0_user = np.zeros((self.num_features, 1), dtype='float16')
        self.mu0_item = np.zeros((self.num_features, 1), dtype='float16')

        self.alpha_user = np.eye(self.num_features, dtype='float16')
        self.alpha_item = np.eye(self.num_features, dtype='float16')

        self.user_features = None
        self.item_features = None

        # num_user, num_item, ratings = build_ml_1m()
        self.matrix = None
        self.train_errors = []
        self.validation_errors = []

        self.mu_item = None
        self.mu_user = None

    def load(self, train, validation):

        self.train = train
        self.validation = validation
        self.num_users = movielens_extractor.get_num_users(self.train)
        self.num_items = movielens_extractor.get_num_items(self.train)
        self.train = train
        self.validation = validation

        self.mean_rating = np.mean(self.train[:, 2])
        self.user_features = 0.1 * np.random.randn(self.num_users, self.num_features)
        self.item_features = 0.1 * np.random.randn(self.num_items, self.num_features)

        # num_user, num_item, ratings = build_ml_1m()
        self.matrix = build_rating_matrix(self.num_users, self.num_items, train)

        self.estimate()

    def estimate(self, iterations=50, tolerance=1e-5):
        last_rmse = None

        # the algorithm will converge, but really slow
        # use MF's initialize latent parameter will be better
        for iteration in xrange(iterations):
            # update item & user parameter
            self.update_item_params()
            self.update_user_params()

            # update item & user_features
            self.udpate_item_features()
            self.update_user_features()

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
            print "iterations: %3d, train RMSE: %.6f, validation RMSE: %.6f " % (iteration + 1, train_rmse, validation_rmse)

            # stop if converge
            # if last_rmse:
            #     if abs(train_rmse - last_rmse) < tolerance:
            #         break

            last_rmse = train_rmse

    def update_item_params(self):
        # N = self.num_items
        X_bar = np.mean(self.item_features, 0)
        # print 'X_bar_prev', X_bar.shape
        X_bar = np.reshape(X_bar, (self.num_features, 1))
        # print 'X_bar', X_bar
        # print 'X_bar', X_bar.shape
        S_bar = np.cov(self.item_features.T)
        # print 'S_bar', S_bar
        # print 'S_bar', S_bar.shape

        # norm_X_bar = X_bar - self.mu_item
        norm_X_bar = self.mu0_item - X_bar
        # TODO: Check why norm_X_bar is here X_bar - mu_item and in the original
        # TODO: implementation is mu_item - X_bar
        # print 'norm_X_bar', norm_X_bar.shape

        WI_post = inv(self.WI_item + self.num_items * S_bar + \
            np.dot(norm_X_bar, norm_X_bar.T) * \
            (self.num_items * self.beta_item) / (self.beta_item + self.num_items))
        # TODO: Check why are they not using inverses (-1) in this code
        # print 'WI_post', WI_post.shape

        # Not sure why we need this...
        WI_post = (WI_post + WI_post.T) / 2.0
        df_post = self.df_item + self.num_items
        # print('df_post', df_post)

        # update alpha_item
        self.alpha_item = wishartrand(df_post, WI_post)
        # print('alpha_item', self.alpha_item)

        # update mu_item
        mu_temp = (self.beta_item * self.mu0_item + self.num_items * X_bar) / \
            (self.beta_item + self.num_items)
        # print "beta_item", self.beta_item
        # print "beta_item", self.beta_item
        # print "num_items", self.num_items
        # print "mu0_item", self.mu0_item
        # print "x_bar", X_bar
        # print mu_temp
        # print "mu_temp", mu_temp.shape
        # print mu_temp
        lam = cholesky(inv(np.dot(self.beta_item + self.num_items, self.alpha_item)))
        lam = lam.T
        # TODO: see why in the original source code lambda is being transposed
        # TODO: and here it isn't
        # print 'lam', lam.shape
        self.mu_item = mu_temp + np.dot(lam, rand.randn(self.num_features, 1))
        # print 'mu_item', self.mu_item.shape

    def update_user_params(self):
        # same as _update_user_params
        # N = self.num_users
        X_bar = np.mean(self.user_features, 0).T
        X_bar = np.reshape(X_bar, (self.num_features, 1))

        # print 'X_bar', X_bar.shape
        S_bar = np.cov(self.user_features.T)
        # print 'S_bar', S_bar.shape

        # norm_X_bar = X_bar - self.mu_user
        norm_X_bar = self.mu0_user - X_bar
        # print 'norm_X_bar', norm_X_bar.shape

        WI_post = inv(self.WI_user + self.num_users * S_bar + \
            np.dot(norm_X_bar, norm_X_bar.T) * \
            (self.num_users * self.beta_user) / (self.beta_user + self.num_users))
        # print 'WI_post', WI_post.shape

        # Not sure why we need this...
        WI_post = (WI_post + WI_post.T) / 2.0
        df_post = self.df_user + self.num_users


        # update alpha_user
        self.alpha_user = wishartrand(df_post, WI_post)
        # print('alpha_user', self.alpha_user)

        # update mu_item
        mu_temp = (self.beta_user * self.mu0_user + self.num_users * X_bar) / \
            (self.beta_user + self.num_users)
        # print 'mu_temp', mu_temp.shape
        lam = cholesky(inv(np.dot(self.beta_user + self.num_users, self.alpha_user)))
        lam = lam.T
        # print 'lam', lam.shape
        self.mu_user = mu_temp + np.dot(lam, rand.randn(self.num_features, 1))
        # print 'mu_user', self.mu_user.shape

    def udpate_item_features(self):
        # Gibbs sampling for item features
        for item_id in xrange(self.num_items):
            users = self.matrix[:, item_id] > 0.0
            # print 'vec', vec.shape
            # if vec.shape[0] == 0:
            #    continue
            features = self.user_features[users, :]
            # print 'features', features.shape
            ratings = self.matrix[users, item_id] - self.mean_rating
            rating_len = len(ratings)
            ratings = np.reshape(ratings, (rating_len, 1))

            # print 'rating', rating.shape
            covar = inv(
                self.alpha_item + self.beta * np.dot(features.T, features))
            # print 'covar', covar.shape
            lam = cholesky(covar)
            lam = lam.T

            temp = self.beta * \
                np.dot(features.T, ratings) + np.dot(
                    self.alpha_item, self.mu_item)
            # print 'temp', temp.shape
            mean = np.dot(covar, temp)
            # print 'mean', mean.shape
            temp_feature = mean + np.dot(lam, rand.randn(self.num_features, 1))
            temp_feature = np.reshape(temp_feature, (self.num_features,))
            self.item_features[item_id, :] = temp_feature
            # print 'item_features', self.item_features.shape

    # def update_user_features(self):
    #     self.matrix = self.matrix.T
    #     # Gibbs sampling for user features
    #     for user_id in xrange(self.num_users):
    #         vec = self.matrix[:, user_id] > 0.0
    #         # print len(vec)
    #         # if vec.shape[0] == 0:
    #         #    continue
    #         # print "item_feature", self.item_features.shape
    #         features = self.item_features[vec, :]
    #         rating = self.matrix[vec, user_id] - self.mean_rating
    #         rating_len = len(rating)
    #         rating = np.reshape(rating, (rating_len, 1))
    #
    #         # print 'rating', rating.shape
    #         covar = inv(
    #             self.alpha_user + self.beta * np.dot(features.T, features))
    #         lam = cholesky(covar)
    #         lam = lam.T
    #         # TODO: see why in the original source code lambda is being transposed
    #         # TODO: and here it isn't
    #
    #         temp = self.beta * \
    #             np.dot(features.T, rating) + np.dot(
    #                 self.alpha_user, self.mu_user)
    #         mean = np.dot(covar, temp)
    #         # print 'mean', mean.shape
    #         temp_feature = mean + np.dot(lam, rand.randn(self.num_features, 1))
    #         temp_feature = np.reshape(temp_feature, (self.num_features,))
    #         self.user_features[user_id, :] = temp_feature
    #
    #     # transpose back
    #     self.matrix = self.matrix.T

    def update_user_features(self):
        self.matrix = self.matrix.T
        # Gibbs sampling for user features
        for user_id in xrange(self.num_users):
            vec = self.matrix[:, user_id] > 0.0
            # print(vec)
            # print len(vec)
            # if vec.shape[0] == 0:
            #    continue
            # print "item_feature", self.item_features.shape
            features = self.item_features[vec, :]
            rating = self.matrix[vec, user_id] - self.mean_rating
            rating_len = len(rating)
            rating = np.reshape(rating, (rating_len, 1))

            # 'rating', rating.shape
            covar = inv(
                self.alpha_user + self.beta * np.dot(features.T, features))
            # lam = cholesky(covar)
            # lam = lam.T
            # TODO: see why in the original source code lambda is being transposed
            # TODO: and here it isn't

            temp = self.beta * \
                np.dot(features.T, rating) + np.dot(
                    self.alpha_user, self.mu_user)
            mean = np.dot(covar, temp)
            lam = cholesky(covar)
            lam = lam.T
            # print 'mean', mean.shape
            temp_feature = mean + np.dot(lam, rand.randn(self.num_features, 1))
            temp_feature = np.reshape(temp_feature, (self.num_features,))
            self.user_features[user_id, :] = temp_feature

        # transpose back
        self.matrix = self.matrix.T

    def predict(self, data):
        u_features = self.user_features[data[:, 0], :]
        i_features = self.item_features[data[:, 1], :]
        preds = np.sum(u_features * i_features, 1) + self.mean_rating

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds


def build_rating_matrix(num_user, num_item, ratings):
    """
    build dense ratings matrix from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print '\nbuild matrix'
    # sparse matrix
    #matrix = sparse.lil_matrix((num_user, num_item))
    # dense matrix
    matrix = np.zeros((num_user, num_item), dtype='int8')
    for item_id in xrange(num_item):
        data = ratings[ratings[:, 1] == item_id]
        if data.shape[0] > 0:
            matrix[data[:, 0], item_id] = data[:, 2]

        if item_id % 1000 == 0:
            print item_id

    return matrix


def RMSE(estimation, truth):
    """Root Mean Square Error"""

    num_sample = len(estimation)

    # sum square error
    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1.0))


def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.arange(1,dim+1)
    foo = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = rand.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))


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
    num_features = 10
    bmf_model = BayesianMatrixFactorization()

    start_time = time.clock()
    bmf_model.load(train, validation)
    end_time = time.clock()
    print "time spend = %.3f" % (end_time - start_time)

    return bmf_model

example()