from scipy.special import gammaln

__author__ = 'fpena'

import numpy as np

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in range(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)


class LatentDirichletAllocation(object):

    def __init__(self, num_topics, alpha=0.1, beta=0.1):
        # self.documents_topics_count = None
        # self.topics_words_count = None
        # self.documents_topics_sum = None
        # self.topics_words_sum = None
        self.alpha = alpha
        self.beta = beta
        # self.num_words = None
        # self.num_docs = None
        self.num_topics = num_topics

    def _initialize(self, matrix):

        self.num_docs, self.num_words = matrix.shape

        # number of times document m and topic z co-occur
        self.documents_topics_count = np.zeros((self.num_docs, self.num_topics))
        # number of times topic z and word w co-occur
        self.topics_words_count = np.zeros((self.num_topics, self.num_words))

        self.documents_topics_sum = np.zeros(self.num_docs)
        self.topics_words_sum = np.zeros(self.num_topics)
        self.topics = {}

        for m in range(self.num_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.num_topics)
                self.documents_topics_count[m,z] += 1
                self.documents_topics_sum[m] += 1
                self.topics_words_count[z,w] += 1
                self.topics_words_sum[z] += 1
                self.topics[(m,i)] = z

    def calculate_p_z(self, d_i, w_i):
        '''
        Calculates the probability that the word w_i, which belongs to document
        d_i is assigned to topic j.

        The formula is p(z_i = j | z_-i, w_i, d_i), which can also be read as
        the probability of topic z_i is equal to j given all the other topics
        assignments, the word and the document. As given by equations 5 from the
        'Finding Scientific Topics' paper written by Griffiths and Steyvers

        :param d_i: the document that contains word w_i
        :param w_i: the given word
        :return:
        '''

        # We calculate the probability mass function of p(z)
        # This includes all the topics (all the values of k)
        left = (self.topics_words_count[:, w_i] + self.beta) /\
               (self.topics_words_sum + self.num_words * self.beta)
        right = (self.documents_topics_count[d_i, :] + self.alpha) /\
                (self.documents_topics_sum[d_i] + self.num_topics * self.alpha)

        # This was giving better results
        # left = (self.topics_words_count[:, w_i] + self.beta) /\
        #        (self.topics_words_sum + self.num_words * self.beta)
        # right = (self.documents_topics_count[d_i, :]) /\
        #         (self.documents_topics_sum[d_i])

        p_z = left * right
        # We normalize the values to obtain the probability
        p_z /= np.sum(p_z)

        # Note: 'p_z' is a multinomial distribution which's length is the number
        #  of topics. It is not a value

        return p_z


    def sample_topic(self, p_z):
        '''
        Sample a new topic from the multinomial distribution p_z and return
        the topic index

        :param p_z:
        :return:
        '''

        return np.random.multinomial(1, p_z).argmax()

    def run(self, matrix, num_cycles):
        """
        Perform inference of the topic model using Gibss Sampling

        :param matrix: a matrix with a count of the words in each document
        :param num_cycles: the number of iterations for the Gibbs Sampling
        routine
        """
        self._initialize(matrix)

        for gibbs_cycle in range(num_cycles):
            for document in range(self.num_docs):
                for i, word in enumerate(word_indices(matrix[document, :])):
                    topic = self.topics[(document, i)]
                    self.documents_topics_count[document, topic] -= 1
                    self.topics_words_count[topic, word] -= 1
                    self.documents_topics_sum[document] -= 1
                    self.topics_words_sum[topic] -= 1

                    p_z = self.calculate_p_z(document, word)
                    new_topic = self.sample_topic(p_z)

                    self.documents_topics_count[document, new_topic] += 1
                    self.topics_words_count[new_topic, word] += 1
                    self.documents_topics_sum[document] += 1
                    self.topics_words_sum[new_topic] += 1
                    self.topics[(document, i)] = new_topic

            yield self.phi()

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.topics_words_count.shape[1]
        num = self.topics_words_count + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def estimate_phi(self):
        """
        Calculates the value of phi, which is a set of T multinomial
        distributions over the W words. This represents p(w|z). In other words,
        phi is a matrix that contains the distribution of words in each topic.

        The value of phi is calculated based on equation 6 from the
        'Finding Scientific Topics' paper written by Griffiths and Steyvers

        :return: a matrix with the words distributions in each topics.
        """
        phi = np.zeros(self.num_topics, self.num_words)
        for topic in range(self.num_topics):
            for word in range(self.num_words):
                phi[topic, word] = \
                    (self.topics_words_count[topic, word] + self.beta) /\
                    (self.topics_words_sum[word] + self.num_words * self.beta)

        return phi

    def estimate_theta(self):
        """
        Calculates the value of theta, which is a set of D multinomial
        distributions over the T topics. This represents p(z). In other words,
        theta is a matrix that contains the distribution of topics in each
        document.

        The value of theta is calculated based on equation 7 from the
        'Finding Scientific Topics' paper written by Griffiths and Steyvers

        :return: a matrix with the topics distributions in each document.
        """
        theta = np.zeros(self.num_docs, self.num_topics)
        for document in range(self.num_docs):
            for word in range(self.num_topics):
                theta[document, word] = \
                    (self.documents_topics_count[document, word] + self.alpha) /\
                    (self.documents_topics_sum[word] + self.num_topics * self.alpha)

        return theta

    def loglikelihood(self):
        return self.loglikelihood_franpena()
        # return self.loglikelihood_mblondiel()
        # return self.loglikelihood_park()

    def loglikelihood_franpena(self):
        """
        Calculates the joint log likelihood log(p(w,z)) to determine how well
        is the algorithm fitting the data. This is the performance metric of
        this algorithm.

        This algorithm calculates p(z,w) = p(w|z) * p(z) based on equations
        2 and 3 from the 'Finding Scientific Topics' paper written by Griffiths
        and Steyvers

        :return: the joint log likelihood value
        """
        likelihood = 0

        # log p(w|z,\beta) --> equation 2
        likelihood += self.num_topics * gammaln(self.num_words * self.beta)
        likelihood -= self.num_topics * self.num_words * gammaln(self.beta)

        for topic in range(self.num_topics):
            likelihood += np.sum(gammaln(self.topics_words_count[topic] + self.beta))
            likelihood -= gammaln(np.sum(self.topics_words_count[topic] + self.num_words * self.beta))

        # log p(z|\alpha) --> equation 3
        likelihood += self.num_docs * gammaln(np.sum(self.alpha) * self.num_topics)
        likelihood -= self.num_docs * self.num_topics * np.sum(gammaln(self.alpha))

        for doc in range(self.num_docs):
            likelihood += np.sum(gammaln(self.documents_topics_count[doc] + self.alpha))
            likelihood -= gammaln(np.sum(self.documents_topics_count[doc] + self.num_topics * self.alpha))

        return likelihood



    def loglikelihood_mblondiel(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.topics_words_count.shape[1]
        n_docs = self.documents_topics_count.shape[0]
        lik = 0

        for z in range(self.num_topics):
            lik += log_multi_beta(self.topics_words_count[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in range(n_docs):
            lik += log_multi_beta(self.documents_topics_count[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.num_topics)

        return lik


    def loglikelihood_park(self):                                        # FIND (JOINT) LOG-LIKELIHOOD VALUE
        l = 0
        for z in range(self.num_topics):                                # log p(w|z,\beta)
            l += gammaln(self.num_words * self.beta)
            l -= self.num_words * gammaln(self.beta)
            l += np.sum(gammaln(self.topics_words_count[z] + self.beta))
            l -= gammaln(np.sum(self.topics_words_count + self.beta))
        for d in range(self.num_docs):                                  # log p(z|\alpha)
            # d = self.indD[doc]
            l += gammaln(np.sum(self.alpha))
            l -= np.sum(gammaln(self.alpha))
            l += np.sum(gammaln(self.documents_topics_count[d] + self.alpha))
            l -= gammaln(np.sum(self.documents_topics_count[d] + self.alpha))
        return l









