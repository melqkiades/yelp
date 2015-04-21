
# from numpy.linalg import inv, cholesky
from numpy.linalg import inv, cholesky
import time
from tripadvisor.fourcity import movielens_extractor
from utils.normal_random import NormalRandom

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
        self.WI_user = np.eye(self.num_features)
        self.beta_user = 2.0
        self.df_user = self.num_features
        self.mu0_user = np.zeros((self.num_features, 1))

        # Inv-Whishart (item features)
        self.WI_item = np.eye(self.num_features)
        self.beta_item = 2.0
        self.df_item = self.num_features
        self.mu0_item = np.zeros((self.num_features, 1))

        # Latent Variables
        self.mu0_user = np.zeros((self.num_features, 1))
        self.mu0_item = np.zeros((self.num_features, 1))

        self.alpha_user = np.eye(self.num_features)
        self.alpha_item = np.eye(self.num_features)

        self.user_features = None
        self.item_features = None
        self.df_post_item = None
        self.df_post_user = None

        self.matrix = None
        self.train_errors = []
        self.validation_errors = []

        self.mu_item = None
        self.mu_user = None
        self.counter_prob = None
        self.probe_rat_all = None
        self.ratings_test = None
        self.results_file = open('/Users/fpena/tmp/bpmf/python-results.txt', 'w')

    def load(self, ratings, train, validation):

        self.train = train
        self.validation = validation
        self.num_users = movielens_extractor.get_num_users(ratings)
        self.num_items = movielens_extractor.get_num_items(ratings)
        self.train = train
        self.validation = validation

        self.mean_rating = np.mean(self.train[:, 2])
        self.ratings_test = np.float64(validation[:, 2])
        self.item_features = 0.1 * NormalRandom.generate_matrix(self.num_items, self.num_features)
        self.user_features = 0.1 * NormalRandom.generate_matrix(self.num_users, self.num_features)

        self.df_post_item = self.df_item + self.num_items
        self.df_post_user = self.df_user + self.num_users

        # num_user, num_item, ratings = build_ml_1m()
        self.matrix = build_rating_matrix(self.num_users, self.num_items, train)
        self.matrix = self.matrix.T
        self.counter_prob = 1
        self.probe_rat_all = self.pred(self.item_features, self.user_features, self.validation, self.mean_rating)

        self.estimate()

    def estimate(self, iterations=50, tolerance=1e-5):
        last_rmse = None

        # the algorithm will converge, but really slow
        # use MF's initialize latent parameter will be better
        for iteration in range(iterations):

            self.results_file.write('Epoch %d\n' % (iteration+1))

            # update item & user parameter
            self.update_item_params()
            self.update_user_params()

            # update item & user_features
            for gibbs_cycle in range(2):
                self.results_file.write('Gibbs cycle %d\n' % (gibbs_cycle+1))
                self.udpate_item_features()
                self.update_user_features()

            # compute RMSE
            # train errors
            train_preds = self.predict(self.train)
            train_rmse = RMSE(train_preds, np.float64(self.train[:, 2]))

            probe_rat = self.pred(self.item_features, self.user_features, self.validation, self.mean_rating)
            self.probe_rat_all = (self.counter_prob*self.probe_rat_all + probe_rat)/(self.counter_prob+1)
            self.counter_prob = self.counter_prob + 1
            temp = np.power((self.ratings_test - self.probe_rat_all), 2)
            average_err = np.sqrt(np.sum(temp)/len(self.validation))

            # validation errors
            validation_preds = self.predict(self.validation)
            validation_rmse = RMSE(
                validation_preds, np.float64(self.validation[:, 2]))
            self.train_errors.append(train_rmse)
            self.validation_errors.append(validation_rmse)

            val = np.sqrt(((validation_preds - self.validation[:, 2]) ** 2).mean())

            # my_rmse = np.sqrt(np.mean(temp))
            # print('Average Test RMSE %3d %.6f \t Other RMSE %.6f ' % (iteration + 1, average_err, my_rmse))

            print ("Epoch: %3d, Average Test RMSE: %.6f , " \
                  "RMSE: %.6f " % (iteration + 1, average_err, val))

            # print "iterations: %3d, train RMSE: %.6f, validation RMSE: %.6f , " \
            #       "Average Test RMSE: %.6f " % (iteration + 1, train_rmse, validation_rmse, average_err)

            # stop if converge
            # if last_rmse:
            #     if abs(train_rmse - last_rmse) < tolerance:
            #         break

            last_rmse = train_rmse

    def update_item_params(self):
        x_bar = np.mean(self.item_features, 0).T
        x_bar = np.reshape(x_bar, (self.num_features, 1))
        S_bar = np.cov(self.item_features.T)
        norm_X_bar = self.mu0_item - x_bar

        WI_post = inv(inv(self.WI_item) + self.num_items * S_bar + \
            np.dot(norm_X_bar, norm_X_bar.T) * \
            (self.num_items * self.beta_item) / (self.beta_item + self.num_items))

        # Not sure why we need this...
        WI_post = (WI_post + WI_post.T) / 2.0

        # update alpha_item
        self.alpha_item = sample_wishart(WI_post, self.df_post_item)

        # update mu_item
        mu_temp = (self.beta_item * self.mu0_item + self.num_items * x_bar) / \
            (self.beta_item + self.num_items)
        lam = cholesky(inv((self.beta_item + self.num_items) * self.alpha_item))
        # lam = lam.T
        self.mu_item = mu_temp + np.dot(lam, NormalRandom.generate_matrix(self.num_features, 1))

        # raise ValueError('AAA')

    def update_user_params(self):
        x_bar = np.mean(self.user_features, 0).T
        x_bar = np.reshape(x_bar, (self.num_features, 1))
        S_bar = np.cov(self.user_features.T)
        norm_X_bar = self.mu0_user - x_bar

        WI_post = inv(inv(self.WI_user) + self.num_users * S_bar + \
            np.dot(norm_X_bar, norm_X_bar.T) * \
            (self.num_users * self.beta_user) / (self.beta_user + self.num_users))

        # Not sure why we need this...
        WI_post = (WI_post + WI_post.T) / 2.0

        # update alpha_user
        self.alpha_user = sample_wishart(WI_post, self.df_post_user)

        # update mu_item
        mu_temp = (self.beta_user * self.mu0_user + self.num_users * x_bar) / \
            (self.beta_user + self.num_users)
        lam = cholesky(inv((self.beta_user + self.num_users) * self.alpha_user))
        self.mu_user = mu_temp + np.dot(lam, NormalRandom.generate_matrix(self.num_features, 1))

    def udpate_item_features(self):
        self.matrix = self.matrix.T
        # Gibbs sampling for item features
        for item_id in range(self.num_items):
            self.results_file.write('Item %d\n' % (item_id+1))
            users = self.matrix[:, item_id] > 0.0
            features = self.user_features[users, :]
            ratings = self.matrix[users, item_id] - self.mean_rating
            rating_len = len(ratings)
            ratings = np.reshape(ratings, (rating_len, 1))

            covar = inv(
                self.alpha_item + self.beta * np.dot(features.T, features))
            lam = cholesky(covar)
            temp = self.beta * \
                np.dot(features.T, ratings) + np.dot(
                    self.alpha_item, self.mu_item)
            mean = np.dot(covar, temp)
            temp_feature = mean + np.dot(lam, NormalRandom.generate_matrix(self.num_features, 1))
            temp_feature = np.reshape(temp_feature, (self.num_features,))
            self.item_features[item_id, :] = temp_feature


    def update_user_features(self):
        self.matrix = self.matrix.T
        # print('matrix', self.matrix.shape, self.matrix[0:10, 0:5])

        # Gibbs sampling for user features
        for user_id in range(self.num_users):
            self.results_file.write('User %d\n' % (user_id+1))
            items = self.matrix[:, user_id] > 0.0
            features = self.item_features[items, :]
            ratings = self.matrix[items, user_id] - self.mean_rating
            rating_len = len(ratings)
            ratings = np.reshape(ratings, (rating_len, 1))

            covar = inv(
                self.alpha_user + self.beta * np.dot(features.T, features))

            temp = self.beta * \
                np.dot(features.T, ratings) + np.dot(
                    self.alpha_user, self.mu_user)
            mean = np.dot(covar, temp)
            lam = cholesky(covar)
            temp_feature = mean + np.dot(lam, NormalRandom.generate_matrix(self.num_features, 1))
            temp_feature = np.reshape(temp_feature, (self.num_features,))
            self.user_features[user_id, :] = temp_feature

        self.results_file.write('user_features \t (%d,%d) \t %16.16f\n' % (self.user_features.shape[0], self.user_features.shape[1], self.user_features[0,0]))
        # transpose back
        # self.matrix = self.matrix.T

    def predict(self, data):
        u_features = self.user_features[data[:, 0], :]
        i_features = self.item_features[data[:, 1], :]
        preds = np.sum(u_features * i_features, 1) + self.mean_rating

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def pred(self, w1_M1_sample,w1_P1_sample,probe_vec,mean_rating):
        users = probe_vec[:, 0]
        items = probe_vec[:, 1]

        pred_out = np.sum(np.multiply(w1_M1_sample[items,:], w1_P1_sample[users,:]),1) + mean_rating

        # print('pred_out', pred_out.shape)
        ff = pred_out > 5
        pred_out[ff] = 5
        ff = pred_out < 1
        pred_out[ff] = 1

        return pred_out


def build_rating_matrix(num_user, num_item, ratings):
    """
    build dense ratings matrix from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print('\nbuild matrix')
    # sparse matrix
    #matrix = sparse.lil_matrix((num_user, num_item))
    # dense matrix
    matrix = np.zeros((num_user, num_item))
    for item_id in range(num_item):
        data = ratings[ratings[:, 1] == item_id]
        if data.shape[0] > 0:
            matrix[data[:, 0], item_id] = data[:, 2]

        if item_id % 1000 == 0:
            print(item_id)

    return matrix


def RMSE(estimation, truth):
    """Root Mean Square Error"""

    num_sample = len(estimation)

    # sum square error
    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1.0))


def sample_wishart(sigma, dof):
    '''
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    '''

    n = sigma.shape[0]

    chol = np.linalg.cholesky(sigma).T

    rnd_matrix = NormalRandom.generate_matrix(dof, n)
    X = np.dot(rnd_matrix, chol)
    W = np.dot(X.T, X)

    return W


def example():
    """simple test and performance measure
    """
    # reviews = movielens_extractor.get_ml_1m_dataset()
    reviews = movielens_extractor.get_ml_100K_dataset()
    ratings = movielens_extractor.reviews_to_numpy_matrix(reviews)
    # suffle_data
    np.random.seed(0)
    np.set_printoptions(precision=16)

    # print(NormalRandom.generate_matrix(1, 10))
    # np.random.shuffle(ratings)

    # split data to training & validation
    train_pct = 0.9
    train_size = int(train_pct * len(ratings))
    train = ratings[:train_size]
    validation = ratings[train_size:]

    # params
    num_features = 10
    bmf_model = BayesianMatrixFactorization()

    start_time = time.clock()
    bmf_model.load(ratings, train, validation)
    end_time = time.clock()
    print("time spent = %.3f" % (end_time - start_time))

    return bmf_model

example()
