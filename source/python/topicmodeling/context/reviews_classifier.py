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

__author__ = 'fpena'


class ReviewsClassifier:

    def __init__(self):
        self.num_features = 2
        self.classifier = LogisticRegression(C=100)
        self.min_values = None
        self.max_values = None

    def train(self, records, reviews=None):

        if reviews is None:
            reviews = []
            for record in records:
                reviews.append(Review(record['text']))

        if len(records) != len(reviews):
            msg = 'The size of the records and reviews arrays must be the same'
            raise ValueError(msg)

        metrics = numpy.zeros((len(reviews), self.num_features))

        for index in range(len(reviews)):
            metrics[index] =\
                review_metrics_extractor.get_review_metrics(reviews[index])

        self.min_values = metrics.min(axis=0)
        self.max_values = metrics.max(axis=0)
        review_metrics_extractor.normalize_matrix_by_columns(
            metrics, self.min_values, self.max_values)

        labels = numpy.array([record['specific'] == 'yes' for record in records])
        self.classifier.fit(metrics, labels)

    def predict(self, reviews):
        metrics = numpy.zeros((len(reviews), self.num_features))
        for index in range(len(reviews)):
            metrics[index] =\
                review_metrics_extractor.get_review_metrics(reviews[index])

        review_metrics_extractor.normalize_matrix_by_columns(
            metrics, self.min_values, self.max_values)
        return self.classifier.predict(metrics)

    def label_json_reviews(self, input_file, output_file, reviews=None):

        records = ETLUtils.load_json_file(input_file)

        if reviews is None:
            reviews = []
            for record in records:
                reviews.append(Review(record['text']))

        if len(records) != len(reviews):
            msg = 'The size of the records and reviews arrays must be the same'
            raise ValueError(msg)
        predicted_classes = self.predict(reviews)

        for record, predicted_class in zip(records, predicted_classes):
            if predicted_class:
                label = 'specific'
            else:
                label = 'generic'

            record['predicted_class'] = label

        ETLUtils.save_json_file(output_file, records)




def plot(my_metrics, my_labels):
    clf = LogisticRegression(C=100)
    clf.fit(my_metrics, my_labels)

    def f_learned(lab):
        return clf.intercept_ + clf.coef_ * lab

    coef = clf.coef_[0]
    intercept = clf.intercept_

    print('coef', coef)
    print('intercept', intercept)

    xvals = numpy.linspace(0, 1.0, 2)
    yvals = -(coef[0] * xvals + intercept[0]) / coef[1]
    plt.plot(xvals, yvals, color='g', label='decision boundary')

    plt.xlabel("log number of words (normalized)")
    plt.ylabel("log number of verbs in past tense (normalized)")
    for outcome, marker, colour in zip([0, 1], "ox", "br"):
        plt.scatter(
            my_metrics[:, 0][my_labels == outcome],
            my_metrics[:, 1][my_labels == outcome], c = colour, marker = marker)
    plt.show()


def main():
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    my_file = my_folder + 'classified_hotel_reviews.json'
    binary_reviews_file = my_folder + 'classified_hotel_reviews.pkl'
    # my_file = my_folder + 'classified_restaurant_reviews.json'
    # binary_reviews_file = my_folder + 'classified_restaurant_reviews.pkl'
    my_records = ETLUtils.load_json_file(my_file)

    # my_reviews = build_reviews(my_records)
    # with open(binary_reviews_file, 'wb') as write_file:
    #     pickle.dump(my_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    with open(binary_reviews_file, 'rb') as read_file:
        my_reviews = pickle.load(read_file)

    num_features = 2

    my_metrics = numpy.zeros((len(my_reviews), num_features))
    for index in range(len(my_reviews)):
        my_metrics[index] =\
            review_metrics_extractor.get_review_metrics(my_reviews[index])
        # print(my_metrics[index])

    review_metrics_extractor.normalize_matrix_by_columns(my_metrics)

    for metric in my_metrics:
        print(metric)

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

    # print(Y)

    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(my_metrics, my_labels)
    # print(knn.predict(my_metrics[0]), my_records[0]['specific'])

    classifiers = [
        DummyClassifier(strategy='most_frequent', random_state=0),
        DummyClassifier(strategy='stratified', random_state=0),
        DummyClassifier(strategy='uniform', random_state=0),
        DummyClassifier(strategy='constant', random_state=0, constant=True),
        LogisticRegression(C=100),
        SVC(C=1.0, kernel='rbf'),
        SVC(C=1.0, kernel='linear'),
        KNeighborsClassifier(n_neighbors=10),
        tree.DecisionTreeClassifier(),
        NuSVC(),
        LinearSVC()
    ]
    scores = [[] for i in range(len(classifiers))]

    Xtrans = my_metrics
    # pca = decomposition.PCA(n_components=2)
    # Xtrans = pca.fit_transform(my_metrics)
    # print(pca.explained_variance_ratio_)
    # lda_inst = lda.LDA()
    # Xtrans = lda_inst.fit_transform(my_metrics, my_labels)
    # print(lda_inst.get_params())
    # mds = manifold.MDS(n_components=2)
    # Xtrans = mds.fit_transform(my_metrics)

    cv = KFold(n=len(my_metrics), n_folds=5)

    for i in range(len(classifiers)):
        for train, test in cv:
            x_train, y_train = Xtrans[train], my_labels[train]
            x_test, y_test = Xtrans[test], my_labels[test]

            clf = classifiers[i]
            clf.fit(x_train, y_train)
            scores[i].append(clf.score(x_test, y_test))

            print(y_test)

            # selector = RFE(clf, n_features_to_select=3)
            # selector = selector.fit(x_train, y_train)
            # print(selector.support_)
            # print(selector.ranking_)

            # model = ExtraTreesClassifier()
            # model.fit(x_train, y_train)
            # display the relative importance of each attribute
            # print(model.feature_importances_)

            # precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
            # print('precision', precision)
            # print('recall', recall)
            # print('thresholds', thresholds)

            # print(clf6.predict_proba(x_test))
            # print(clf6.coef_)

            # print(classification_report(y_test, clf.predict_proba(x_test)[:,1]>0.8,
            #                             target_names=['generic', 'specific']))


    for classifier, score in zip(classifiers, scores):
        print("Mean(scores)=%.5f\tStddev(scores)=%.5f" % (numpy.mean(score), numpy.std(score)))
    # for classifier, score in zip(classifiers, scores):
    #     print("Mean(scores)=%.5f\tStddev(scores)=%.5f\t%s" % (numpy.mean(score), numpy.std(score), classifier))

    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(my_metrics, my_labels)
    # from sklearn.externals.six import StringIO
    # import pydot
    # dot_data = StringIO()
    # sklearn.tree.export_graphviz(clf, out_file=dot_data)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("/Users/fpena/tmp/iris.pdf")

    plot(my_metrics, my_labels)


def test():
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    my_file = my_folder + 'classified_restaurant_reviews.json'
    binary_reviews_file = my_folder + 'classified_restaurant_reviews.pkl'
    my_training_records = ETLUtils.load_json_file(my_file)

    # my_reviews = build_reviews(my_records)
    # with open(binary_reviews_file, 'wb') as write_file:
    #     pickle.dump(my_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    with open(binary_reviews_file, 'rb') as read_file:
        my_training_reviews = pickle.load(read_file)

    classifier = ReviewsClassifier()
    classifier.train(my_training_records, my_training_reviews)

    my_input_records_file = my_folder + 'yelp_training_set_review_restaurants_shuffled.json'
    my_input_reviews_file = my_folder + 'reviews_restaurant_shuffled.pkl'
    my_output_records_file = my_folder + 'yelp_training_set_review_restaurants_shuffled_tagged.json'

    with open(my_input_reviews_file, 'rb') as read_file:
        my_input_reviews = pickle.load(read_file)

    classifier.label_json_reviews(
        my_input_records_file, my_output_records_file, my_input_reviews)


# start = time.time()
# # main()
# test()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
