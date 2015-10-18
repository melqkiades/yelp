from collections import Counter
import cPickle as pickle
import string
import time
import math
import nltk
import numpy
import sklearn
from sklearn import decomposition, lda, manifold
from sklearn.cross_validation import KFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import tree
import matplotlib.pyplot as plt
from etl import ETLUtils
from topicmodeling.context import review_metrics_extractor

__author__ = 'fpena'


NUM_FEATURES = 2


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

    plt.xlabel("Ln number of words (normalized)")
    plt.ylabel("Ln number of verbs in past tense (normalized)")
    for outcome, marker, colour in zip([0, 1], "ox", "br"):
        plt.scatter(
            my_metrics[:, 0][my_labels == outcome],
            my_metrics[:, 1][my_labels == outcome], c = colour, marker = marker)
    plt.show()


def main():
    my_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    # my_file = my_folder + 'classified_hotel_reviews.json'
    # binary_reviews_file = my_folder + 'classified_hotel_reviews.pkl'
    my_file = my_folder + 'classified_restaurant_reviews.json'
    binary_reviews_file = my_folder + 'classified_restaurant_reviews.pkl'
    my_records = ETLUtils.load_json_file(my_file)

    print(my_records[10])

    # my_reviews = build_reviews(my_records)
    # with open(binary_reviews_file, 'wb') as write_file:
    #     pickle.dump(my_reviews, write_file, pickle.HIGHEST_PROTOCOL)

    with open(binary_reviews_file, 'rb') as read_file:
        my_reviews = pickle.load(read_file)

    my_metrics = numpy.zeros((len(my_reviews), NUM_FEATURES))
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
        # scores.append([])
        for train, test in cv:
            x_train, y_train = Xtrans[train], my_labels[train]
            x_test, y_test = Xtrans[test], my_labels[test]

            clf = classifiers[i]
            clf.fit(x_train, y_train)
            scores[i].append(clf.score(x_test, y_test))

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







start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)







# from nltk.corpus import wordnet as wn
# animate = x = filter(lambda s:s.lexname() in {'noun.time'}, wn.all_synsets('n'))
# animate = {lemma.name() for s in animate for lemma in s.lemmas()}
# print(animate)
# print(time.time())
# for i in range(1000000):
#     boolean = 'yesterday' in animate
# print(time.time())


# start = time.time()
#
# review1 = 'Sheraton Desert Oasis is a great value in the late spring/early summer to get away for a few days and relax. \n\nIt is part of the Starwood Hotel Chain, but this is one of their first forays in to the recent boom in Hotel chains opening time share residences for sale under their brands. So this was meant to be property that was sold in timeshares, which some have. mostly for winter weeks for cold snow birds.\n\nBut in the low season, which is summer, you can get a room for around $100. And what do you get? You get a nicely appointed, decorated model home, about a 700 square foot 1 bedroom apartment. It is repleat with 2 gas fireplaces, full kitchen with all wares and utensils, living room, patio with table and chairs, his/her bathroom sinks, a marbled tiled shower, a king size bed and a heart shaped large jacuzzi bathtub in the bedroom. \n\nIt has a full social and activities calender, so it is a good place to take the kids. There is a large clover shaped pool, with multiple water fountains, and a large rocklike island structure with caves where the jacuzzi is located. There are about 75 chaise lounges and several tables, so there is usually one available for you. There is also a poolside grill for food and beverages. \n\nYou are not far from the restaurants and nightlife of North Scottsdale, near Kierland Commons Mall, Desert Ridge Mall, 101 Mall, and about 6 other strip and big box shopping centers. You can buy food and easily eat in to save some money. The famed Tournament Players Club (TPC) gold course is close by, as well as the more exclusive and expensive Fairmont Princess hotel. \n\nThis is more of a vacation home than an exclusive resort with people waiting on you hand and foot. But it is well maintained, nicely appointed, and a relaxing getaway. Peak season (winter) rates are $325 and up, so if you come in early summer, you get a bargain and a tan by the pool.'
# review2 = 'Known as \"The Jewel of the Desert\" and the \"Grand Dame,\" this beautiful hotel is the only one in the world with a Frank Lloyd Wright-influenced design.  The architect was a man named Albert Chase McArthur, who had studied under Frank Lloyd Wright.  Hence, the main restaurant\'s name is \"Frank and Albert\'s.\"\n\nCelebrating its 80th birthday this year, this unique art-deco resort is rich in history.  Dozens of presidents have golfed here, Marilyn Monroe splashed around in one of the 8 on-site pools, and many more celebrities and industry magnates have called this their temporary home.\n\nWhen visiting here, it is important to note that on-site parking is costly.  A self parking fee of $12 is imposed even on its guests, and the self parking lot is often located some distance from the hotel entrance.  For those not wanting to bother with parking the car themselves, the price is$27 to valet.  The package I purchased luckily had the self-parking included, but the car was delivered to the valet nonetheless.  I was sure to speak to a hotel manager about how ridiculous it is to ask a hotel guest to pay for parking, and he happily credited our room for the valet fee.\n\nSo, my first experience was our dinner at \"Frank and Albert\'s\", formerly known as the \"Biltmore Grill.\"  Local purveyors provide the ingredients for nearly all of its organic dishes.  From Queen Creek Olive Oil to Shriner\'s Sausage, Mesa Ranch Goat Cheese to Biltmore grown Rosemary, one can taste the difference.  The menu provided a range of selections from salads, pizzas, and flatbreads, to rotisserie and classic comfort foods.  The wood stone baked filet of salmon was flavorful, tender, and full of tender feelings. I recommend their \"bloody rose\" cocktail.  I snicker as I write these words, because bloody marys are my least favorite cocktail.  However, this unique blend is made with crop organic tomato vodka and juice, fresh biltmore grown rosemary and spices.  Service was attentive and timely, waiter was very knowledgeable, and we were very impressed with the quality and flavor of the food.  \n\nNext, it was on to the newly added wing of the resort, the Ocatilla.  The Ocatilla is essentially a new boutique hotel situated on the Biltmore property.  It has its own pool, a concierge, and an executive lounge which offers daily complimentary breakfasts, a light evening fare, beer and wine, desserts, as well as coffee and soft drinks.Our package included a room on the third floor of this location, as well as access to its executive lounge.  \n\nAlthough our stomachs were full from dinner, we made room for chocolate covered cheesecake, choc. covered strawberries, fudge brownies, tarts, and chocolate chunk cookies.  We swallowed it all down with some prosecco, and headed up to our room.\n\nThe room was at least 500sq feet, and I believe that the rest of the hote has fairly sizeable rooms, as well.  The decor was a blend of art deco and southwestern.  The bathroom with its granite countertops, boasted a tub with sea salts and candle, in which we submerged ourselves comfortably up to our chins.  Two plush robes were there for us to lounge around in.  A snack bar was provided, although a package of nuts and dried fruit was $7.  Biggest disappointment was that that there was no wetbar.  I always look forward to these in any hotel room, especially to use them for storing leftovers from dinner.  The king bed was etremely comfortable, with its pillow-top mattress and feather pillows, as well as the high thread count sheets.  Silky soft...\n\nWe enjoyed peace and quiet all night, and were not disturbed in the morning by housekeeping.\n\nThe next day were were early to rise to enjoy the lavish continental breakfast spread in the lounge.  Granola with local pecans and yogurt parfait croissans with spinach and boursin, ham and cheese, small danishes, assortmnt of fresh fruits and juices gave us the perfect start to our day.  We grabbed some bottled waters and cokes and headed to the pool.  We had an appointment at the spa an hour later, for seaweed wraps and massages, and it was wonderful, the service excellent and very personalized.\n\nRecommendations:\ngo to biltmore\'s website directly to book a room. you will find incredible packages. our package was $200 and included dinner for two, lunch for two, 50% off at the spa, and access to the executive lounge at the ocatilla.\n\nif you live locally, they will shuttle you - saves the parking headache.\n\nAbout Ocatilla:\nbasically an upgrade with your stay. For an additional $75-$100/day, these are the following amenities you will receive:\ncomplimentary beverages, wine, beer, breakfast, light evening fare, and desserts in the lounge\nbusiness center (computers, internet, printers)\nWii rentals \npressing service and shoe shine\ntherapeutic turn down bath\nneck and shoulder massages every thursday\npersonal concierge'
#
# classifier = ReviewsClassifier()
# print(classifier.get_time_words(review1))
# print(classifier.get_time_words(review2))
#
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

# nltk.download()

# yesterday = sentiwordnet.senti_synset('yesterday.n.01')
# print(yesterday)

# happy = sentiwordnet.senti_synsets('time')
# all_happy = sentiwordnet.all_senti_synsets()
#
# for synset in happy:
#     print(synset)




# evenly sampled time at 200ms intervals
t = numpy.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()

import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)



