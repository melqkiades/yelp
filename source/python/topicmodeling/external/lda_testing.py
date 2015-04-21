__author__ = 'fpena'


import numpy as np
import lda
import lda.datasets


def run():
    # document-term matrix
    X = lda.datasets.load_reuters()
    print("type(X): {}".format(type(X)))
    print("shape: {}\n".format(X.shape))

    # the vocab
    vocab = lda.datasets.load_reuters_vocab()
    print("type(vocab): {}".format(type(vocab)))
    print("len(vocab): {}\n".format(len(vocab)))

    # titles for each story
    titles = lda.datasets.load_reuters_titles()
    print("type(titles): {}".format(type(titles)))
    print("len(titles): {}\n".format(len(titles)))

    doc_id = 0
    word_id = 3117

    print("doc id: {} word id: {}".format(doc_id, word_id))
    print("-- count: {}".format(X[doc_id, word_id]))
    print("-- word : {}".format(vocab[word_id]))
    print("-- doc : {}".format(titles[doc_id]))

    model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    model.fit(X)
    topic_word = model.topic_word_

    print("type(topic_word): {}".format(type(topic_word)))
    print("shape: {}".format(topic_word.shape))

    for n in range(5):
        sum_pr = sum(topic_word[n,:])
        print("topic: {} sum: {}".format(n, sum_pr))

    n = 5
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))

    for n in range(5):
        sum_pr = sum(doc_topic[n,:])
        print("document: {} sum: {}".format(n, sum_pr))

    for n in range(10):
        topic_most_pr = doc_topic[n].argmax()
        print("doc: {} topic: {}\n{}...".format(n,
                                                topic_most_pr,
                                                titles[n][:50]))

reuters_dataset = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

print('Dataset shape', reuters_dataset.shape)
print(reuters_dataset[0].shape)

print('Vocab shape', len(vocab))
print(vocab[0])

print('Titles shape', len(titles))
print(titles[0])
print(titles[1])
print(titles[100])






for word in reuters_dataset[0]:
    if word > 1:
        print(word)
