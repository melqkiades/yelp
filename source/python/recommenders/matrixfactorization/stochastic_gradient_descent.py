import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import time
from recommenders.base_recommender import BaseRecommender
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity import movielens_extractor

__author__ = 'fpena'


class StochasticGradientDescent(BaseRecommender):
    def __init__(self, num_features):
        super(StochasticGradientDescent, self).__init__(
            'StochasticGradientDescentRecommender', None, None)
        self.num_features = num_features
        self.reviews = None
        self.user_ids = None
        self.item_ids = None
        self.p = None
        self.q = None
        self.data_matrix = None
        self.num_users = None
        self.num_items = None
        self.learning_rate = 0.001
        self.regularization = 0.02
        self.n_p = None
        self.n_q = None
        self.user_index_map = None
        self.item_index_map = None

    def load(self, reviews):
        self.reviews = reviews
        self.user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        self.item_ids = extractor.get_groupby_list(self.reviews, 'offering_id')
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self.data_matrix = create_matrix(self.reviews)
        self.user_index_map = build_user_index_map(reviews)
        self.item_index_map = build_item_index_map(reviews)

        R = create_matrix(self.reviews).todense()

        R = np.array(R)

        N = len(R)
        M = len(R[0])
        K = 2

        P = np.random.rand(N, K)
        Q = np.random.rand(M, K)
        # P.fill(0.1)
        # Q.fill(0.1)

        self.n_p, self.n_q = matrix_factorization(R, P, Q, K)

    def predict_rating(self, user_id, item_id):
        # return np.dot(self.p[user_id - 1, :], self.q[:, item_id - 1])
        return np.dot(self.n_p[self.user_index_map[user_id]],
                      self.n_q[self.item_index_map[item_id]])


test_reviews = [
    {'user_id': 'A1', 'offering_id': 1, 'overall_rating': 4.0},
    # {'user_id': 'A1', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'A1', 'offering_id': 2, 'overall_rating': 2.0},
    {'user_id': 'A1', 'offering_id': 3, 'overall_rating': 3.0},
    {'user_id': 'A2', 'offering_id': 1, 'overall_rating': 4.0},
    {'user_id': 'A2', 'offering_id': 3, 'overall_rating': 2.0},
    {'user_id': 'A3', 'offering_id': 1, 'overall_rating': 5.0}
]

reviews_matrix_5 = [
    {'user_id': 'U1', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U1', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U1', 'offering_id': 4, 'overall_rating': 7.0},
    # {'user_id': 'U1', 'offering_id': 5, 'overall_rating': 4.0},
    {'user_id': 'U2', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U2', 'offering_id': 4, 'overall_rating': 7.0},
    {'user_id': 'U2', 'offering_id': 5, 'overall_rating': 9.0},
    {'user_id': 'U3', 'offering_id': 1, 'overall_rating': 5.0},
    {'user_id': 'U3', 'offering_id': 2, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 3, 'overall_rating': 5.0},
    {'user_id': 'U3', 'offering_id': 4, 'overall_rating': 7.0},
    {'user_id': 'U3', 'offering_id': 5, 'overall_rating': 9.0},
    {'user_id': 'U4', 'offering_id': 1, 'overall_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 2, 'overall_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 3, 'overall_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 4, 'overall_rating': 6.0},
    {'user_id': 'U4', 'offering_id': 5, 'overall_rating': 5.0},
    {'user_id': 'U5', 'offering_id': 1, 'overall_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 2, 'overall_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 3, 'overall_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 4, 'overall_rating': 6.0},
    {'user_id': 'U5', 'offering_id': 5, 'overall_rating': 5.0},
]


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    @INPUT:
        R     : a matrix to be factorized, dimension N x M
        P     : an initial matrix of dimension N x K
        Q     : an initial matrix of dimension M x K
        K     : the number of latent features
        steps : the maximum number of steps to perform the optimisation
        alpha : the learning rate
        beta  : the regularization parameter
    @OUTPUT:
        the final matrices P and Q
    """
    Q = Q.T
    for step in range(steps):

        start_time = time.time()

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        # eR = np.dot(P,Q)
        # e = 0
        # for i in xrange(len(R)):
        #     for j in xrange(len(R[i])):
        #         if R[i][j] > 0:
        #             e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
        #             for k in xrange(K):
        #                 e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        # print('Step', step, 'Error', e)
        # if e < 0.001:
        #     break

        end_time = time.time() - start_time
        # print('Cycle', step, 'Time', end_time)
    return P, Q.T


def create_matrix(reviews):
    user_index_map = build_user_index_map(reviews)
    item_index_map = build_item_index_map(reviews)
    num_users = len(user_index_map)
    num_items = len(item_index_map)

    reviews_matrix = sparse.lil_matrix((num_users, num_items), dtype=np.float64)

    for review in reviews:
        user_index = user_index_map[review['user_id']]
        item_index = item_index_map[review['offering_id']]
        reviews_matrix[user_index, item_index] = \
            np.float64(review['overall_rating'])

    return reviews_matrix.tocsr()


def build_user_index_map(reviews):
    user_ids = extractor.get_groupby_list(reviews, 'user_id')
    user_index_map = {}
    index = 0

    for user_id in user_ids:
        user_index_map[user_id] = index
        index += 1

    return user_index_map


def build_item_index_map(reviews):
    user_ids = extractor.get_groupby_list(reviews, 'offering_id')
    item_index_map = {}
    index = 0

    for user_id in user_ids:
        item_index_map[user_id] = index
        index += 1

    return item_index_map


def test():
    R = [
        # [5, 3, 0, 1],
        # [4, 0, 0, 1],
        # [1, 1, 0, 5],
        # [1, 0, 0, 4],
        # [0, 1, 5, 4],
        [5.,  7.,  5.,  7.,  0.],
        [5.,  7.,  5.,  7.,  9.],
        [5.,  7.,  5.,  7.,  9.],
        [6.,  6.,  6.,  6.,  5.],
        [6.,  6.,  6.,  6.,  5.]
    ]

    my_reviews = movielens_extractor.clean_reviews(movielens_extractor.get_ml_100K_dataset())
    R = create_matrix(my_reviews).todense()

    R = np.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    # P.fill(0.1)
    # Q.fill(0.1)

    nP, nQ = matrix_factorization(R, P, Q, K)

    # print(nP)
    # print(nQ)
    predicted = np.dot(nP[0], nQ[4])
    # predicted = np.dot(nP[0, :], nQ[4, :])
    # print('Predicted', predicted)



# my_reviews = movielens_extractor.get_ml_100K_dataset()
# my_reviews = movielens_extractor.clean_reviews(movielens_extractor.get_ml_100K_dataset())

# sgd = StochasticGradientDescent(2)

# sgd.load(reviews_matrix_5)

# print(build_user_index_map(test_reviews))
# print(build_item_index_map(test_reviews))
#
# print(create_matrix(test_reviews))
# print(create_matrix(test_reviews).todense())
# print(create_matrix(test_reviews).nonzero())
#
# print(create_matrix(reviews_matrix_5))
# print(create_matrix(reviews_matrix_5).todense())
# print(create_matrix(reviews_matrix_5).nonzero())
# print(sgd.predict(4, 0))

# start_time = time.time()
# test()
# end_time = time.time() - start_time
# print('Total time', end_time)