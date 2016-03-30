# from topicmodeling.hiddenfactortopics.topic_corpus import TopicCorpus

__author__ = 'fpena'

# import sys
# sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')

# from topicmodeling.hiddenfactortopics.topic_corpus import TopicCorpus

# latent_reg = 0
# lambda_param = 0.1
# num_topics = 5
# # file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-short.votes'
# file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-shuffled.votes'
# # file_name = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-short-shuffled.votes'
# corpus = Corpus()
# corpus.load_data(file_name, 0)
# topic_corpus = TopicCorpus(corpus, num_topics, latent_reg, lambda_param)
# topic_corpus.train(2, 50)

# my_file = '/Users/fpena/UCC/Thesis/external/McAuley2013/code_RecSys13/Arts.votes'
# my_corpus = Corpus(my_file, 0)
# my_corpus.process_reviews(my_file, 0)
# my_corpus.build_votes(my_file, 0)

import sys
sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')

import cPickle as pickle
import time
import numpy
from sklearn.cross_validation import KFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import tree
import matplotlib.pyplot as plt

from etl import ETLUtils
from topicmodeling.context import review_metrics_extractor
from topicmodeling.context.review import Review


def main():
    item_type = 'hotel'
    # item_type = 'restaurant'
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    my_file = my_folder + 'classified_' + item_type + '_reviews.json'
    binary_reviews_file = my_folder + 'classified_' + item_type + '_reviews.pkl'
    my_records = ETLUtils.load_json_file(my_file)

    with open(binary_reviews_file, 'rb') as read_file:
        my_reviews = pickle.load(read_file)

    num_features = 2

    my_metrics = numpy.zeros((len(my_reviews), num_features))
    for index in range(len(my_reviews)):
        my_metrics[index] =\
            review_metrics_extractor.get_review_metrics(my_reviews[index])

    review_metrics_extractor.normalize_matrix_by_columns(my_metrics)

    count_specific = 0
    count_generic = 0
    for record in my_records:

        if record['specific'] == 'yes':
            count_specific += 1

        if record['specific'] == 'no':
            count_generic += 1

    print('count_specific: %d' % count_specific)
    print('count_generic: %d' % count_generic)
    print('specific percentage: %f%%' % (float(count_specific)/len(my_records)))
    print('generic percentage: %f%%' % (float(count_generic)/len(my_records)))

    my_labels = numpy.array([record['specific'] == 'yes' for record in my_records])

    classifiers = [
        DummyClassifier(strategy='most_frequent', random_state=0),
        DummyClassifier(strategy='stratified', random_state=0),
        DummyClassifier(strategy='uniform', random_state=0),
        # DummyClassifier(strategy='constant', random_state=0, constant=True),
        LogisticRegression(C=100),
        SVC(C=1.0, kernel='rbf'),
        SVC(C=1.0, kernel='linear'),
        KNeighborsClassifier(n_neighbors=10),
        tree.DecisionTreeClassifier(),
        NuSVC(),
        LinearSVC()
    ]
    scores = [[] for _ in range(len(classifiers))]

    Xtrans = my_metrics

    cv = KFold(n=len(my_metrics), n_folds=5)

    for i in range(len(classifiers)):
        for train, test in cv:
            x_train, y_train = Xtrans[train], my_labels[train]
            x_test, y_test = Xtrans[test], my_labels[test]

            clf = classifiers[i]
            clf.fit(x_train, y_train)
            scores[i].append(clf.score(x_test, y_test))

    for classifier, score in zip(classifiers, scores):
        print("Mean(scores)=%.5f\tStddev(scores)=%.5f" % (numpy.mean(score), numpy.std(score)))

    plot(my_metrics, my_labels)


def plot(my_metrics, my_labels):
    clf = LogisticRegression(C=100)
    clf.fit(my_metrics, my_labels)

    def f_learned(lab):
        return clf.intercept_ + clf.coef_ * lab

    coef = clf.coef_[0]
    intercept = clf.intercept_

    # print('coef', coef)
    # print('intercept', intercept)

    xvals = numpy.linspace(0, 1.0, 2)
    yvals = -(coef[0] * xvals + intercept[0]) / coef[1]
    plt.plot(xvals, yvals, color='g')

    plt.xlabel("log number of words (normalized)")
    plt.ylabel("log number of verbs in past tense (normalized)")
    my_legends = ['Generic reviews', 'Specific reviews']
    for outcome, marker, colour, legend in zip([0, 1], "ox", "br", my_legends):
        plt.scatter(
            my_metrics[:, 0][my_labels == outcome],
            my_metrics[:, 1][my_labels == outcome], c = colour, marker = marker, label=legend)
    # plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
    plt.legend(loc='lower center', numpoints=1, ncol=3)
    # plt.legend(numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0, 1, 1))
    plt.show()

main()



#
# logging.config.dictConfig({
#     'version': 1,
#     'disable_existing_loggers': False,  # this fixes the problem
#
#     'formatters': {
#         'standard': {
#             'format': '%(asctime)s [%(levelname)s]\t %(name)s: %(message)s'
#             # 'format': '[%(levelname)s] %(message)s [%(module)s %(funcName)s %(lineno)d]'
#         },
#     },
#     'handlers': {
#         'default': {
#             'level':'DEBUG',
#             'class':'logging.StreamHandler',
#             'formatter': 'standard',
#         },
#     },
#     'loggers': {
#         '': {
#             'handlers': ['default'],
#             'level': 'DEBUG',
#             'propagate': True
#         }
#     }
# })
#
# logger = logging.getLogger(__name__)
#
# def test():
#     logger.debug('hola')

# logger.debug("Debug")
# logger.info("Total time = %f seconds" % total_time)
