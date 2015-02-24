
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

        # num_user, num_item, ratings = build_ml_1m()
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

        # print('Num items', self.num_items)
        # print('Num users', self.num_users)
        print('Mean rating', self.mean_rating)

        # print('Item features')
        # print(self.item_features[0:2])
        # print('User features')
        # print(self.user_features[0:2])

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

            # print('mu_item', self.mu_item)
            # print('mu_user', self.mu_user)
            # print('item_features', self.item_features.shape, self.item_features[0:5, :])  # Lost
            # print('user_features', self.user_features.shape, self.user_features[1:5])  # Completely lost
            # print('probe_rat_all', self.probe_rat_all.shape)

            probe_rat = self.pred(self.item_features, self.user_features, self.validation, self.mean_rating)
            self.probe_rat_all = (self.counter_prob*self.probe_rat_all + probe_rat)/(self.counter_prob+1)
            self.counter_prob = self.counter_prob + 1
            temp = np.power((self.ratings_test - self.probe_rat_all), 2)
            average_err = np.sqrt(np.sum(temp)/len(self.validation))

            # print('Average Test RMSE %3d %.6f ' % (iteration + 1, err))



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

        # print('WI_post:', WI_post)

        # update alpha_item
        # self.alpha_item = wishartrand(self.df_post_item, WI_post)
        self.alpha_item = sample_wishart(WI_post, self.df_post_item)

        # update mu_item
        mu_temp = (self.beta_item * self.mu0_item + self.num_items * x_bar) / \
            (self.beta_item + self.num_items)
        lam = cholesky(inv((self.beta_item + self.num_items) * self.alpha_item))
        # lam = lam.T
        self.mu_item = mu_temp + np.dot(lam, NormalRandom.generate_matrix(self.num_features, 1))
        # print('item_features', self.item_features.shape)
        # print('x_bar:', x_bar.shape, x_bar)
        # print('S_bar:', S_bar.shape, S_bar)
        # print('alpha_item:', self.alpha_item.shape, self.alpha_item)
        # print('norm_X_bar:', norm_X_bar.shape, norm_X_bar)
        # print('WI_post:', WI_post.shape, WI_post)
        # print('df_post_item:', self.df_post_item.shape, self.df_post_item)
        # print('lam', lam)
        # print('mu_item', self.mu_item)

        self.results_file.write('alpha_item \t (%d,%d) \t %16.16f\n' % (self.alpha_item.shape[0], self.alpha_item.shape[1], self.alpha_item[0,0]))
        self.results_file.write('mu_item \t (%d,%d) \t %16.16f\n' % (self.mu_item.shape[0], self.mu_item.shape[1], self.mu_item[0,0]))

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
        # self.alpha_user = wishartrand(self.df_post_user, WI_post)
        self.alpha_user = sample_wishart(WI_post, self.df_post_user)

        # update mu_item
        mu_temp = (self.beta_user * self.mu0_user + self.num_users * x_bar) / \
            (self.beta_user + self.num_users)
        lam = cholesky(inv((self.beta_user + self.num_users) * self.alpha_user))
        self.mu_user = mu_temp + np.dot(lam, NormalRandom.generate_matrix(self.num_features, 1))
        # print('mu_user', self.mu_user)
        self.results_file.write('alpha_user \t (%d,%d) \t %16.16f\n' % (self.alpha_user.shape[0], self.alpha_user.shape[1], self.alpha_user[0,0]))
        self.results_file.write('mu_user \t (%d,%d) \t %16.16f\n' % (self.mu_user.shape[0], self.mu_user.shape[1], self.mu_user[0,0]))

    def udpate_item_features(self):
        self.matrix = self.matrix.T
        # Gibbs sampling for item features
        for item_id in range(self.num_items):
            self.results_file.write('Item %d\n' % (item_id+1))
            users = self.matrix[:, item_id] > 0.0
            features = self.user_features[users, :]
            ratings = self.matrix[users, item_id] - self.mean_rating
            # print('ratings00', ratings[0:5])
            rating_len = len(ratings)
            ratings = np.reshape(ratings, (rating_len, 1))

            covar = inv(
                self.alpha_item + self.beta * np.dot(features.T, features))
            lam = cholesky(covar)
            # print('lam', lam)
            temp = self.beta * \
                np.dot(features.T, ratings) + np.dot(
                    self.alpha_item, self.mu_item)
            mean = np.dot(covar, temp)
            # print('mean', mean.shape, mean)
            temp_feature = mean + np.dot(lam, NormalRandom.generate_matrix(self.num_features, 1))
            # print('temp_feature', temp_feature.shape, temp_feature)
            temp_feature = np.reshape(temp_feature, (self.num_features,))
            self.item_features[item_id, :] = temp_feature
            # self.results_file.write('alpha_item \t (%d,%d) \t %16.16f\n' % (self.alpha_item.shape[0], self.alpha_item.shape[1], self.alpha_item[0,0]))
            # self.results_file.write('beta \t %16.16f\n' % (self.beta))
            # if len(ratings):
            #     self.results_file.write('ratings \t (%d,%d) \t %16.16f\n' % (ratings.shape[0], ratings.shape[1], ratings[0,0]))
            # else:
            #     self.results_file.write('No ratings\n')
            # if len(features):
            #     self.results_file.write('features \t (%d,%d) \t %16.16f\n' % (features.shape[0], features.shape[1], features[0,0]))
            # else:
            #     self.results_file.write('No features\n')
            # self.results_file.write('covar \t (%d,%d) \t %16.16f\n' % (covar.shape[0], covar.shape[1], covar[0,0]))

            if item_id == 0:
                # print('users', users.shape, np.sum(users), users)
                # print('users', users.shape, np.sum(users), users)
                # print('features', features.shape, features[0:5, :])
                # print('alpha_item', self.alpha_item.shape, self.alpha_item)
                # print('ratings', ratings[0:5, :])
                # print('beta', self.beta)
                # print('item_features', self.item_features.shape)
                # print('items sum', np.sum(items))
                # print('items', items.shape, items)
                # print('covar', covar.shape, covar)
                # print('lam', lam.shape, lam)
                # print('mean_m', mean.shape, mean)
                # print('temp_feature', temp_feature.shape, temp_feature)
                # print('user_features', self.user_features.shape, self.user_features[0:10, 0:5])
                # print('item_features', self.item_features.shape, self.item_features[0:5, :])
                pass

            # self.results_file.write('features \t (%d,%d) \t %16.16f\n' % (features.shape[0], features.shape[1], features[0,0]))
            # self.results_file.write('features \t (%d,%d) \t %16.16f\n' % (ratings.shape[0], ratings.shape[1], ratings[0,0]))

        self.results_file.write('item_features \t (%d,%d) \t %16.16f\n' % (self.item_features.shape[0], self.item_features.shape[1], self.item_features[0,0]))

        # print('item_features', self.item_features.shape, self.item_features[0:5, :])


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
            # print('lam', lam.shape, lam)
            temp_feature = mean + np.dot(lam, NormalRandom.generate_matrix(self.num_features, 1))
            temp_feature = np.reshape(temp_feature, (self.num_features,))
            self.user_features[user_id, :] = temp_feature

            # if len(ratings):
            #     self.results_file.write('ratings \t (%d,%d) \t %16.16f\n' % (ratings.shape[0], ratings.shape[1], ratings[0,0]))
            # else:
            #     self.results_file.write('No ratings\n')
            # if len(features):
            #     self.results_file.write('features \t (%d,%d) \t %16.16f\n' % (features.shape[0], features.shape[1], features[0,0]))
            # else:
            #     self.results_file.write('No features\n')
            # self.results_file.write('covar \t (%d,%d) \t %16.16f\n' % (covar.shape[0], covar.shape[1], covar[0,0]))

            if user_id == 0:
                # print('alpha_user', self.alpha_user)
                # print('alpha_beta', self.beta)
                # print('item_features', self.item_features.shape)
                # print('items sum', np.sum(items))
                # print('items', items.shape, items)
                # print('features', features.shape, features[0:10, 0:5])
                # print('covar', covar.shape, covar)
                # print('lam', lam.shape, lam)
                # print('temp_feature', temp_feature.shape, temp_feature)
                # print('user_features', self.user_features.shape, self.user_features[0:10, 0:5])
                pass

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
        ratings = np.float64(probe_vec[:, 2])

        # print('w1_M1_sample', w1_M1_sample.shape)
        # print('w1_P1_sample', w1_P1_sample.shape)

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


# def wishartrand(nu, phi):
#     dim = phi.shape[0]
#     chol = cholesky(phi)
#     #nu = nu+dim - 1
#     #nu = nu + 1 - np.arange(1,dim+1)
#     foo = np.zeros((dim,dim))
#
#     for i in range(dim):
#         for j in range(i+1):
#             if i == j:
#                 foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
#             else:
#                 foo[i,j]  = rand.normal(0,1)
#     return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

def sample_wishart(sigma, dof):
    '''
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    '''

    n = sigma.shape[0]
    # print('n:', n)

    chol = np.linalg.cholesky(sigma).T

    rnd_matrix = NormalRandom.generate_matrix(dof, n)
    X = np.dot(rnd_matrix, chol)

    # print('chol:', chol.shape, chol)
    # print('X:', X.shape, X[0:5, :])

    # # use matlab's heuristic for choosing between the two different sampling schemes
    # if (dof <= 81+n) and (dof == round(dof)):
    #     print('Entre al if!')
    #     # direct
    #     X = np.dot(chol,np.random.normal(size=(n,dof)))
    # else:
    #     print('Entre al else!')
    #     A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n),size=n)))
    #     A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
    #     X = np.dot(chol,A)

    W = np.dot(X.T, X)
    # print('W:', W.shape, W)

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
np.random.seed(0)
# l = NormalRandom.generate_matrix(2, 3)
# print(l)
# print(l[1, :])

# Z = np.zeros((500,))
# Z[0] = 3
# Z[1] = 1
# Za = (Z - np.mean(Z)) / np.std(Z)
# print Za[0]

# bb = 1
# aaa1 = 1.0079376598851518
#
# for i in xrange(10000):
#     bb *= aaa1
#
# print('bb', bb)

np.set_printoptions(precision=16)

# my_wi_post = np.matrix('5.64372912923683e-02  -3.36583758861754e-04  -1.44243213424373e-03; -3.36583758861754e-04   5.92934537067332e-02  -1.23138096616869e-03; -1.44243213424373e-03  -1.23138096616869e-03   5.72054735028673e-02')
# my_df_mpost_item = 168
# my_wishart = sample_wishart(my_wi_post, my_df_mpost_item)
# my_z = np.matrix('0.0950634836997276   0.4289768905320054   0.5242622295661438;'
#         '0.2325142230424487  -0.2393522319282718   0.4454650665919888;'
#         '-0.0359572268173166   0.2315594055077436   0.0942324583112557;'
#         '-0.0245212217343758   0.3542594401317679   0.0276539643248734;'
#         '0.0289057667965716   0.1851390404627895   0.0751490311743118')
# my_w = np.dot(my_z.T, my_z)

# print(my_wi_post)
# print(my_df_mpost_item)
# print(my_wishart.dtype)
# print(my_wishart)
# print(my_z)
# print(my_w)


arr = [1, 2, 3, 4, 5, 6]
# print(np.random.permutation(6))
# np.random.shuffle(arr)
# print(arr)

# from utils.normal_random import NormalRandom
# import numpy as np
# import sys
# np.random.seed(0)
# np.set_printoptions(precision=16)
# my_rnd_matrix = NormalRandom.generate_matrix(3, 500000)
# my_dot = np.dot(my_rnd_matrix, my_rnd_matrix.T)
# # #
# print(sys.version_info)
# print(my_dot)
# print(my_dot[0,0] - 499404)



