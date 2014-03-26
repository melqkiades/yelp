import numpy as np
import time
import sys
from netflix.data_loader import DataLoader

__author__ = 'Anamaria Todor' # https://github.com/absynthe


class AlternateSVD:

    NUM_FEATURES = 20
    FEATURE_INIT_VALUE = 0.1


    def __init__(self, review_matrix):
        self.model = review_matrix
        self.user_feature_matrix = self.create_user_feature_matrix()
        self.movie_feature_matrix = self.create_movie_feature_matrix()

        # OK
    def create_user_feature_matrix(self):
        """
        Creates a user feature matrix of size NUM_FEATURES X NUM_USERS
        with all cells initialized to FEATURE_INIT_VALUE

        :rtype : numpy matrix
        :return: a matrix of size NUM_FEATURES X NUM_USERS
        with all cells initialized to FEATURE_INIT_VALUE
        """
        num_users = self.model.shape[0]
        user_feature_matrix = np.empty((num_users, self.NUM_FEATURES))
        user_feature_matrix[:] = self.FEATURE_INIT_VALUE
        return user_feature_matrix

    # OK
    def create_movie_feature_matrix(self):
        """
        Creates a user feature matrix of size NUM_FEATURES X NUM_MOVIES
        with all cells initialized to FEATURE_INIT_VALUE

        :rtype : numpy matrix
        :return: a matrix of size NUM_FEATURES X NUM_MOVIES
        with all cells initialized to FEATURE_INIT_VALUE
        """
        num_movies = self.model.shape[1]
        movie_feature_matrix = np.empty((num_movies, self.NUM_FEATURES))
        movie_feature_matrix[:] = self.FEATURE_INIT_VALUE
        return movie_feature_matrix

    def learn(self):
        K = 20
        steps = 120
        learning_rate = 0.001
        regularization = 0.25
        num_ratings = np.count_nonzero(self.model.toarray().ravel())


        '''
        N = self.model.shape[0] #no of users
        M = self.model.shape[1] #no of items
        #self.p = np.random.rand(N, K)
        #self.q = np.random.rand(M, K)
        self.p = self.user_feature_matrix
        self.q = self.movie_feature_matrix
        rows,cols = self.model.nonzero()
        for step in xrange(steps):
            squared_error = 0
            for u, i in zip(rows,cols):
                e = self.model[u, i] - np.dot(self.p[u, :], self.q[i, :]) #calculate error for gradient
                p_temp = learning_rate * ( e * self.q[i,:] - regularization * self.p[u,:])
                self.q[i,:]+= learning_rate * ( e * self.p[u,:] - regularization * self.q[i,:])
                self.p[u,:] += p_temp
                squared_error += e
            rmse = (squared_error / num_ratings) ** 0.5
            print(rmse)
        '''

        #self.p = np.random.rand(N, K)
        #self.q = np.random.rand(M, K)
        num_users = self.model.shape[0]
        num_movies = self.model.shape[1]
        self.p=np.empty([num_users,K])
        self.q=np.empty([num_movies,K])
        self.p.fill(0.1)
        self.q.fill(0.1)
        self.q=self.q.T

        rows,cols = self.model.nonzero()
        average_time = 0.0
        for step in xrange(steps):
            start_time = time.time()
            squared_error = 0
            for u, i in zip(rows,cols):
                e_ui=self.model[u,i]-np.dot(self.p[u,:],self.q[:,i]) #calculate error for gradient
                for k in xrange(K):
                    # adjust P and Q based on error gradient
                    temp =self.p[u,k] + learning_rate * (e_ui * self.q[k,i] - regularization * self.p[u,k])
                    self.q[k,i]=self.q[k,i] + learning_rate * (e_ui * self.p[u,k] - regularization * self.q[k,i])
                    self.p[u,k]= temp
                squared_error += e_ui
            rmse = (squared_error / num_ratings) ** 0.5
            print('RMSE' + str(rmse))
            average_time +=time.time() - start_time
        sys.stdout.flush()
        print "One plain step took on average" + str(average_time/steps), " seconds"

movielens_file_path = 'E:/UCC/Thesis/datasets/ml-100k/u1.base'
my_reviews = DataLoader.create_review_matrix(movielens_file_path)
matrix_factorizer = AlternateSVD(my_reviews)
matrix_factorizer.learn()
