import os
import shutil

import numpy as np
import scipy.misc

from topicmodeling.external.lda_gibbs_mblondel import LdaSampler, sample_index


N_TOPICS = 10
DOCUMENT_LENGTH = 100
FOLDER = "topicimg"

def vertical_topic(width, topic_index, document_length):
    """
    Generate a topic whose words form a vertical bar.
    """
    m = np.zeros((width, width))
    m[:, topic_index] = int(document_length / width)
    return m.flatten()

def horizontal_topic(width, topic_index, document_length):
    """
    Generate a topic whose words form a horizontal bar.
    """
    m = np.zeros((width, width))
    m[topic_index, :] = int(document_length / width)
    return m.flatten()

def save_document_image(filename, doc, zoom=2):
    """
    Save document as an image.
    doc must be a square matrix
    """
    height, width = doc.shape
    zoom = np.ones((width*zoom, width*zoom))
    # imsave scales pixels between 0 and 255 automatically
    scipy.misc.imsave(filename, np.kron(doc, zoom))

def gen_word_distribution(n_topics, document_length):
    """
    Generate a word distribution for each of the n_topics.
    """
    width = int(n_topics / 2)
    vocab_size = width ** 2
    m = np.zeros((n_topics, vocab_size))

    for k in range(width):
        m[k,:] = vertical_topic(width, k, document_length)

    for k in range(width):
        m[k+width,:] = horizontal_topic(width, k, document_length)

    m /= m.sum(axis=1)[:, np.newaxis] # turn counts into probabilities

    return m

def gen_document(word_dist, n_topics, vocab_size, length=DOCUMENT_LENGTH, alpha=0.1):
    """
    Generate a document:
        1) Sample topic proportions from the Dirichlet distribution.
        2) Sample a topic index from the Multinomial with the topic
        proportions from 1).
        3) Sample a word from the Multinomial corresponding to the topic
        index from 2).
        4) Go to 2) if need another word.
    """
    theta = np.random.mtrand.dirichlet([alpha] * n_topics)
    v = np.zeros(vocab_size)
    for n in range(length):
        z = sample_index(theta)
        w = sample_index(word_dist[z,:])
        v[w] += 1
    return v

def gen_documents(word_dist, n_topics, vocab_size, n=500):
    """
    Generate a document-term matrix.
    """
    m = np.zeros((n, vocab_size))
    for i in range(n):
        m[i, :] = gen_document(word_dist, n_topics, vocab_size)
    return m

if os.path.exists(FOLDER):
    shutil.rmtree(FOLDER)
os.mkdir(FOLDER)

width = N_TOPICS / 2
vocab_size = width ** 2
word_dist = gen_word_distribution(N_TOPICS, DOCUMENT_LENGTH)
matrix = gen_documents(word_dist, N_TOPICS, vocab_size)
sampler = LdaSampler(N_TOPICS)

for it, phi in enumerate(sampler.run(matrix)):
    print("Iteration", it)
    print("Likelihood", sampler.loglikelihood())

    if it % 5 == 0:
        for z in range(N_TOPICS):
            save_document_image("topicimg/topic%d-%d.png" % (it,z),
                                phi[z,:].reshape(width,-1))
