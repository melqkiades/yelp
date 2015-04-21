import os
import re
from pickle import load

from termcolor import colored
import numpy as np
import lda

from topicmodeling.latent_dirichlet_allocation import LatentDirichletAllocation


def info(s):
    print(colored(s, 'yellow'))

def readDocs(directory):
    docs = []
    pattern = re.compile('[\W_]+')
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # print colored(filename, 'red')
            with open(root + '/' + filename) as f:
                header = True
                content = []
                for line in f:
                    if not header:
                        words = [pattern.sub('', w.lower()) for w in line.split()]
                        content.extend(words)
                    elif line.startswith('Lines: '):
                        header = False

        docs.append(content)
    return docs

def preprocess(directory):
    info('Reading corpus')
    docs = readDocs(directory)
    stopwords = load(open('stopwords.pickle'))

    info('Building vocab')
    vocab = set()
    for doc in docs:
        for w in doc:
            if len(w) > 1 and w not in stopwords:
                vocab.add(w)

    vocab = list(vocab)
    lookupvocab = dict([(v, k) for (k, v) in enumerate(vocab)])

    info('Building BOW representation')
    m = np.zeros((len(docs), len(vocab)))
    for d, doc in enumerate(docs):
        for w in doc:
            if len(w) > 1 and w not in stopwords:
                m[d, lookupvocab[w]] += 1
    return m, vocab

def discoverTopics(n = 20):
    matrix, vocab = preprocess('/Users/fpena/tmp/20_newsgroups')
    # matrix, vocab = preprocess('../data/toy2')
    # sampler = LdaSampler(n)
    sampler = LatentDirichletAllocation(n)

    info('Starting!')
    for it, phi in enumerate(sampler.run(matrix, 100)):
        print(colored("Iteration %s" % it, 'yellow'))
        print("Likelihood", sampler.loglikelihood())

        # for topicNum in range(n):
        #     s = colored(topicNum, 'green')
        #     # s = topicNum
        #     words = [(proba, w) for (w, proba) in enumerate(phi[topicNum, :]) if proba > 0]
        #     words = sorted(words, reverse = True)
        #     for i in range(10):
        #         proba, w = words[i]
        #         s += ' ' + vocab[w]
        #     print(s)

    lda_sampler = lda.LDA(n, 100)
    lda_sampler._fit(matrix)
    print("LDA likelihood", lda_sampler.loglikelihood())

discoverTopics()

# my_dir = '/Users/fpena/tmp/20_newsgroups'
# my_docs = readDocs(my_dir)
# my_matrix, my_vocab = preprocess(my_dir)
#
# print('Num docs:', len(my_docs))
# print('Matrix shape', my_matrix.shape)
# print(my_matrix[0].shape)
# print('Vocab shape', len(my_vocab))
#
# # print(my_docs[0])
# print(my_docs[1])
#
# for word in my_matrix[1]:
#     if word > 0:
#         print(word)
