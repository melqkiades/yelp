import math
import sys
from etl import ETLUtils
from topicmodeling.hiddenfactortopics.corpus import Corpus
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend('/Users/fpena/UCC/Thesis/projects/yelp/source/python/topicmodeling/hiddenfactortopics')


__author__ = 'fpena'

import numpy as np

class TopicCorpus:

    # TODO: Create a list with all the users and all the items as strings

    def __init__(self, corpus, num_topics, latent_reg, lambda_param):

        """

        :type corpus: Corpus
        :param corpus:
        :type num_topics: int
        :param num_topics:
        :type latent_reg: float
        :param latent_reg:
        :type lambda_param: float
        :param lambda_param:
        """
        np.random.seed(0)
        self.corpus = corpus

        # Votes from the training, validation, and test sets
        self.train_votes = None
        self.valid_votes = None
        self.test_votes = None

        self.best_valid_predictions = {}

        self.votes_per_item = []  # Vector of votes for each item
        self.votes_per_user = []  # Vector of votes for each user
        self.train_votes_per_item = []  # Same as above, but only votes from the training set
        self.train_votes_per_user = []

        # Model parameters
        self.alpha = None  # Offset parameter
        self.kappa = 1.0  # "peakiness" parameter
        self.beta_user = None  # User offset parameters
        self.beta_item = None  # Item offset parameters
        self.gamma_user = None  # User latent factors
        self.gamma_item = None  # Item latent factors

        # Contiguous version of all parameters, i.e., a flat vector containing
        # all parameters in order (useful for lbfgs)
        self.w = None

        self.topic_words = None  # Weights each word in each topic
        self.background_words = None  # "background" weight, so that each word has average weight zero across all topics

        # Latent variables
        self.word_topics = {}

        # Counters
        self.item_topic_counts = None  # How many times does each topic occur for each product?
        self.item_words = None  # Number of words in each "document"
        self.topic_counts = None  # How many times does each topic occur?
        self.word_topic_counts = None  # How many times does this topic occur for this word?
        self.total_words = None  # How many words are there?

        self.nw = None
        self.num_topics = num_topics

        self.latent_reg = latent_reg  # Regularization parameter
        self.lambda_param = lambda_param  # Learning rate

        self.n_training_per_user = {}  # Number of training items for each user
        self.n_training_per_item = {}  # Number of training items for each item

        self.n_users = None  # Number of users
        self.n_items = None  # Number of items
        self.n_words = None  # Number of words
        self.user_list = None
        self.item_list = None

        self.d_alpha = None
        self.d_kappa = None
        self.d_beta_user = None
        self.d_beta_item = None
        self.d_gamma_user = None
        self.d_gamma_item = None
        self.d_topic_words = None

        self._initialize()

    def _initialize(self):
        self.n_users = self.corpus.num_users
        self.n_items = self.corpus.num_items
        self.n_words = self.corpus.num_words

        print('n_users', self.n_users)
        print('n_items', self.n_items)
        print('n_words', self.n_words)

        self.beta_user = np.zeros(self.n_users)
        self.beta_item = np.zeros(self.n_items)
        self.votes_per_user = [[] for _ in range(self.n_users)]
        self.votes_per_item = [[] for _ in range(self.n_items)]
        self.train_votes_per_user = [[] for _ in range(self.n_users)]
        self.train_votes_per_item = [[] for _ in range(self.n_items)]
        self.gamma_user = np.zeros((self.n_users, self.num_topics))
        self.gamma_item = np.zeros((self.n_items, self.num_topics))

        for vote in self.corpus.vote_list:
            self.votes_per_user[vote.user].append(vote)

        for user in range(self.n_users):
            for vote in self.votes_per_user[user]:
                # print_vote(vote)
                # print('item', vote.item)
                self.votes_per_item[vote.item].append(vote)

        self.split_data()

        # total number of parameters
        self.nw = 1 + 1 + (self.num_topics + 1) * \
                          (self.n_users + self.n_items) + \
                  self.num_topics * self.n_words

        self.alpha = self._calculate_average_rating(self.train_votes)

        print('init alpha', self.alpha)

        train, valid, test, test_ste = self.valid_test_error()
        print("Error w/ offset term only (train/valid/test) = %f/%f/%f (%f)" %
              (train, valid, test, test_ste))

        self._calculate_user_item_offsets()
        train, valid, test, test_ste = self.valid_test_error()
        print("Error w/ offset and bias (train/valid/test) = %f/%f/%f (%f)" %
              (train, valid, test, test_ste))

        # Actually the model works better if we initialize none of these terms
        if self.lambda_param > 0:
            self.alpha = 0
            self.beta_user = np.zeros(self.n_users)
            self.beta_item = np.zeros(self.n_items)

        self.word_topic_counts = np.zeros((self.n_words, self.num_topics))

        self.generate_random_topic_assignments()
        self.init_background_word_frequency()
        self.init_gamma_matrices()

        self.normalize_word_weights()
        if self.lambda_param > 0:
            self.update_topics()
        self.kappa = 1

    def split_data(self):
        """
        We split the data into training and validation
        NOTE: Beware that there could be users/items that don't appear in the
        training set in the test set

        """
        self.train_votes, validation_test_votes = ETLUtils.split_train_test(
            self.corpus.vote_list, split=0.8, shuffle_data=False)
        self.valid_votes, self.test_votes = ETLUtils.split_train_test(
            validation_test_votes, split=0.5, shuffle_data=False)

        for vote in self.train_votes:
            user = vote.user
            item = vote.item
            if user not in self.n_training_per_user:
                self.n_training_per_user[user] = 0
            if item not in self.n_training_per_item:
                self.n_training_per_item[item] = 0
            self.n_training_per_user[user] += 1
            self.n_training_per_item[item] += 1
            self.train_votes_per_user[user].append(vote)
            self.train_votes_per_item[item].append(vote)

    def init_gamma_matrices(self):
        if self.lambda_param == 0:
            for user in range(self.n_users):
                if user not in self.n_training_per_user:
                    continue
                for k in range(self.num_topics):
                    self.gamma_user[user][k] = np.random.rand()
            for item in range(self.n_items):
                if item not in self.n_training_per_item:
                    continue
                for k in range(self.num_topics):
                    self.gamma_user[item][k] = np.random.rand()
        else:
            self.topic_words = np.zeros((self.n_words, self.num_topics))

    def init_background_word_frequency(self):
        # Initialize the background word frequency
        self.total_words = 0
        self.background_words = np.zeros(self.n_words)

        for vote in self.train_votes:
            for word in vote.word_list:
                self.total_words += 1
                self.background_words[word] += 1

        self.background_words /= self.total_words

        # for word in range(self.n_words):
        #     print("background_words[%d]\t= %f" % (word, self.background_words[word]))

        # print("total words = %d" % self.total_words)
        # print("background words = %d" % self.background_words)

    def generate_random_topic_assignments(self):
        # Generate random topic assignments
        self.topic_counts = np.zeros(self.num_topics)
        self.item_topic_counts = np.zeros((self.n_items, self.num_topics))
        self.item_words = np.zeros(self.n_items)

        for vote in self.train_votes:
            self.word_topics[vote] = np.zeros(self.n_words)
            self.item_words[vote.item] += len(vote.word_list)

            print_vote(vote)

            # print("item_words[%d]\t = %d" % (vote.item, self.item_words[vote.item]))

            for wp in range(len(vote.word_list)):
                wi = vote.word_list[wp]
                t = np.random.random() * self.num_topics

                # print("wi = %d\tt = %d" % (wi, t))

                self.word_topics[vote][wp] = t
                self.item_topic_counts[vote.item][t] += 1
                self.word_topic_counts[wi][t] += 1
                self.topic_counts[t] += 1

        for item in range(self.n_items):
            for k in range(self.num_topics):
                print("item_topic_counts[%d][%d]\t = %d" % (item, k, self.item_topic_counts[item][k]))

    def _split_data_set(self):
        pass

    @staticmethod
    def _calculate_average_rating(vote_list):

        average_rating = 0
        for vote in vote_list:
            # print_vote(vote)
            average_rating += vote.rating
        return average_rating / len(vote_list)

    def _calculate_user_item_offsets(self):
        for vote in self.train_votes:
            self.beta_user[vote.user] += vote.rating - self.alpha
            self.beta_item[vote.item] += vote.rating - self.alpha
        for user in range(self.n_users):
            self.beta_user[user] /= len(self.votes_per_user[user])
            # print('user', user)
            # print('beta user %d = %f' % (user, self.beta_user[user]))
        for item in range(self.n_items):
            self.beta_item[item] /= len(self.votes_per_item[item])
            # print('beta item %d = %f\tsize = %d' % (item, self.beta_item[item], len(self.votes_per_item[item])))

    def prediction(self, vote):
        """
        Predict a particular rating given the current parameter values

        :type vote: Vote
        """
        user = vote.user
        item = vote.item
        res = self.alpha + self.beta_user[user] + self.beta_item[item]

        for k in range(self.num_topics):
            res += self.gamma_user[user][k] * self.gamma_item[item][k]

        return res

    # TODO: Finish this method
    def dl(self):
        """
        Derivative of the energy function

        """

        self.d_alpha = 0
        self.d_kappa = 0
        self.d_beta_user = np.zeros(self.n_users)
        self.d_beta_item = np.zeros(self.n_items)
        self.d_gamma_user = np.zeros((self.n_users, self.num_topics))
        self.d_gamma_item = np.zeros((self.n_items, self.num_topics))
        self.d_topic_words = np.zeros((self.n_words, self.num_topics))

        for user in range(self.n_users):
            for vote in self.train_votes_per_user[user]:
                p = self.prediction(vote)
                pred_error = 2 * (p - vote.rating)

                self.d_alpha += pred_error
                self.d_beta_user[user] += pred_error

                for k in range(self.num_topics):
                    self.d_gamma_user[user][k] -=\
                        pred_error * self.gamma_item[vote.item][k]

        for item in range(self.n_items):
            for vote in self.train_votes_per_item[item]:
                p = self.prediction(vote)
                pred_error = 2 * (p - vote.rating)

                self.d_beta_item[item] += pred_error

                for k in range(self.num_topics):
                    self.d_gamma_item[item][k] +=\
                        pred_error * self.gamma_user[vote.user][k]

        # dk = 0
        for item in range(self.n_items):
            tZ = self.topic_z(item)

            for k in range(self.num_topics):
                # print("lambda = %f" % self.lambda_param)
                # print("item_topic_counts[%d][%d] = %d" % (item, k, self.item_topic_counts[item][k]))
                # print('%f\t%d\t%d\t%f\t%f\t%f' %
                #       (self.lambda_param, self.item_topic_counts[item][k],
                #        self.item_words[item], self.kappa,
                #        self.gamma_item[item][k], tZ))
                q = - self.lambda_param * (self.item_topic_counts[item][k] - self.item_words[item] * math.exp(self.kappa * self.gamma_item[item][k]) / tZ)
                print('q = %f' % q)
                # print('kappa', self.kappa)
                self.d_gamma_item[item][k] += self.kappa * q
                self.d_kappa += self.gamma_item[item][k] * q

        # Add the derivative of the regularizer
        if self.latent_reg > 0:
            for user in range(self.n_users):
                for k in range(self.num_topics):
                    self.d_gamma_user[user][k] +=\
                        self.latent_reg * 2 * self.d_gamma_user[user][k]
            for item in range(self.n_items):
                for k in range(self.num_topics):
                    self.d_gamma_item[item][k] +=\
                        self.latent_reg * 2 * self.d_gamma_item[item][k]

        wZ = self.word_z()

        for word in range(self.n_words):
            for k in range(self.num_topics):
                twC = self.word_topic_counts[word][k]
                ex = math.exp(self.background_words[word] +
                              self.topic_words[word][k])
                self.d_topic_words[word][k] +=\
                    -self.lambda_param *\
                    (twC - self.topic_counts[k] * ex / wZ[k])

        print("d_alpha = %f" % self.d_alpha)
        print("d_kappa = %f" % self.d_kappa)

        for item in range(self.n_items):
            for k in range(self.num_topics):
                print("item_topic_counts[%d][%d]\t = %d" % (item, k, self.item_topic_counts[item][k]))

    def update_gradient(self):

        learning_rate = 0.00001

        self.alpha -= learning_rate * self.d_alpha
        self.kappa -= learning_rate * self.d_kappa
        self.beta_user -= learning_rate * self.d_beta_user
        self.beta_item -= learning_rate * self.d_beta_item
        self.gamma_user -= learning_rate * self.d_gamma_user
        self.gamma_item -= learning_rate * self.d_gamma_item
        self.topic_words -= learning_rate * self.d_topic_words

    def train(self, em_iterations, grad_iterations):
        """

        :type em_iterations: int
        :type grad_iterations: int
        """
        best_valid = float("inf")

        for emi in range(em_iterations):

            for gi in range(grad_iterations):
                # evaluate
                self.dl()
                self.update_gradient()

            if self.lambda_param > 0:
                # print(self.gamma_user)
                self.update_topics()
                self.normalize_word_weights()
                self.top_words()

            train, valid, test, test_ste = self.valid_test_error()
            print("Error (train/valid/test) = %f/%f/%f (%f)\n" %
                  (train, valid, test, test_ste))

            if valid < best_valid:
                best_valid = valid
                for vote in self.corpus.vote_list:
                    self.best_valid_predictions[vote] = self.prediction(vote)



    def lsq(self):
        """
        Compute the energy according to the least-squares criterion

        :return:
        """
        res = 0

        for vote in self.train_votes:
            res += (self.prediction(vote) - vote.rating) ** 2

        for item in range(self.n_items):
            tZ = self.topic_z(item)
            lZ = math.log(tZ)

            for k in range(self.num_topics):
                res += -self.lambda_param * self.item_topic_counts[item][k] * \
                       (self.kappa * self.gamma_item[item][k] - lZ)

        # Add the regularizer to the energy
        if self.latent_reg > 0:
            for user in range(self.n_users):
                for k in range(self.num_topics):
                    res += self.latent_reg * self.gamma_user[user][k] ** 2
            for item in range(self.n_items):
                for k in range(self.num_topics):
                    res += self.latent_reg * self.gamma_item[item][k] ** 2

        wZ = self.word_z()

        # TODO: Pythonize this, remove the for and put it in one line
        for k in range(self.num_topics):
            lZ = math.log(wZ[k])
            for word in range(self.n_words):
                res += -self.lambda_param * self.word_topic_counts[word][k] * \
                       (self.background_words[word] + self.topic_words[word][k] - lZ)

        return res

    def valid_test_error(self):
        """
        Compute the validation and test error (and testing standard error)

        :type train: float
        :type valid: float
        :type test: float
        :type test_ste: float
        """
        train = 0
        valid = 0
        test = 0
        test_ste = 0

        error_vs_training_user = {}
        error_vs_training_item = {}

        for train_vote in self.train_votes:
            train += (self.prediction(train_vote) - train_vote.rating) ** 2
        for valid_vote in self.valid_votes:
            valid += (self.prediction(valid_vote) - valid_vote.rating) ** 2
        for test_vote in self.test_votes:
            err = (self.prediction(test_vote) - test_vote.rating) ** 2
            test += err
            test_ste += err ** 2

            if test_vote.user in self.n_training_per_user:
                nu = self.n_training_per_user[test_vote.user]
                if nu not in error_vs_training_user:
                    error_vs_training_user[nu] = []
                error_vs_training_user[nu].append(err)

            if test_vote.item in self.n_training_per_item:
                ni = self.n_training_per_item[test_vote.item]
                if ni not in error_vs_training_item:
                    error_vs_training_item[ni] = []
                error_vs_training_item[ni].append(err)

        # Standard error
        for key, value in error_vs_training_item.iteritems():
            if key > 100:
                continue
            av, var = self.average_var(value)

        train /= len(self.train_votes)
        valid /= len(self.valid_votes)
        test /= len(self.test_votes)
        test_ste /= len(self.test_votes)
        test_ste = math.sqrt((test_ste - test ** 2) / len(self.test_votes))

        return train, valid, test, test_ste

    def average_var(self, values):
        """

        :type values: list[float]
        :param values:
        """
        sq = 0
        av = 0

        for value in values:
            av += value
            sq += value ** 2

        var = sq - av ** 2

        return av, var

    def normalize_word_weights(self):
        """
        Subtract averages from word weights so that each word has average weight
        zero across all topics (the remaining weight is stored in
        "self.background_words")

        """
        for w in range(self.n_words):
            av = 0
            for k in range(self.num_topics):
                av += self.topic_words[w][k]
            av /= self.num_topics
            for k in range(self.num_topics):
                self.topic_words[w][k] -= av
            self.background_words[w] -= av

    def save(self, model_path, prediction_path):
        """

        :type model_path: str
        :type prediction_path: str
        """
        pass

    def get_g(self, g, alpha, kappa, beta_user, beta_item, gamma_user,
              gamma_item, topic_words, init):
        """

        :type g: float
        :type alpha: float
        :type kappa: float
        :type beta_user: float
        :type beta_item: float
        :type gamma_user: float
        :type gamma_item: float
        :type topic_words: float
        :type init: bool
        """
        pass

    # def clear_g(self):
    #     pass

    def word_z(self):
        """
        Compute normalization constants for all K topics
        Look at equation 9 in the paper
        """

        res = np.zeros(self.num_topics)

        for k in range(self.num_topics):
            for w in range(self.n_words):
                res[k] += math.exp(self.background_words[w] + self.topic_words[w][k])

        return res

    def topic_z(self, item):
        """
        Compute normalization constant for a particular item
        Look at equation 4 in the paper

        :type item: int
        :type res: float
        """
        res = 0
        # print('kappa', self.kappa)
        for k in range(self.num_topics):
            # print('gamma_item', self.gamma_item[item][k])
            res += math.exp(self.kappa * self.gamma_item[item][k])

        return res

    # def gradient_descent(self):
    #
    # def train(self, em_iterations, grad_iterations):

    def update_topics(self):
        """
        Update topic assignments for each word, this is done by sampling

        """
        for vote in self.train_votes:
            item = vote.item
            topics = self.word_topics[vote]

            for wp in range(len(vote.word_list)):
                # For each word position
                wi = vote.word_list[wp]
                topic_scores = np.zeros(self.num_topics)
                topic_total = 0

                for k in range(self.num_topics):
                    topic_scores[k] = math.exp(
                        self.kappa * self.gamma_item[item][k] +
                        self.background_words[wi] + self.topic_words[wi][k])
                    topic_total += topic_scores[k]

                # TODO: Pythonize
                for k in range(self.num_topics):
                    topic_scores[k] /= topic_total

                new_topic = 0
                x = np.random.random()
                while True:
                    x -= topic_scores[new_topic]
                    if x < 0:
                        break
                    new_topic += 1

                if new_topic != topics[wp]:
                    # Update topic counts if the topic for this word position
                    # changed
                    t = topics[wp]
                    self.word_topic_counts[wi][t] -= 1
                    self.word_topic_counts[wi][new_topic] += 1
                    self.topic_counts[t] -= 1
                    self.topic_counts[new_topic] += 1
                    self.item_topic_counts[item][t] -= 1
                    self.item_topic_counts[item][new_topic] += 1
                    topics[wp] = new_topic

    def top_words(self):
        """
        Print out the top words for each topic

        """
        pass

    @staticmethod
    def main():
        latent_reg = 0
        lambda_param = 0.1
        num_topics = 5
        # file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-short.votes'
        file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-shuffled.votes'
        # file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-short-shuffled.votes'
        corpus = Corpus()
        corpus.load_data(file_name, 0)
        topic_corpus = TopicCorpus(corpus, num_topics, latent_reg, lambda_param)
        topic_corpus.train(2, 50)


def print_vote(vote):
    """

    :type vote: Vote
    :param vote:
    """
    print("Vote: user = %d\titem = %d\t rating = %f" %
          (vote.user, vote.item, vote.rating))

    # for word in vote.word_list:
    #     print(word)

TopicCorpus.main()

# np.random.seed(0)
# random.seed(0)
# print(random.random())
# print(np.random.rand())
# print(np.random.rand())
# print(np.random.rand())
# print(np.random.rand())
# print(np.random.rand())
# print(np.random.rand())
# print(np.random.rand())
