import time
from collections import Counter

import pandas
from nltk.metrics import agreement
from sklearn.metrics import cohen_kappa_score

from etl import ETLUtils
from utils.constants import Constants


def load_data(file_name):
    records = ETLUtils.load_json_file(file_name)
    data_frame = pandas.DataFrame.from_records(records)

    column = 'review_type'
    # column = 'specific'

    # print(data_frame.describe())
    # print(data_frame.head())
    # data_frame = data_frame['specific']
    # print(data_frame.groupby(column)[column].count())
    # reviews = list(data_frame['text'])
    values = list(data_frame[column])
    values = [value.encode('ascii', 'ignore') for value in values]
    # print(reviews)
    print(values)

    return records


def compare_records():
    data_folder = '/Users/fpena/UCC/Thesis/datasets/context/manuallyLabeledReviews/'

    business_type = Constants.ITEM_TYPE
    file_name = data_folder + '%s_%s_reviews.json'

    labelers = [
        # 'francisco',
        'diego',
        'mesut',
        'rohit',
    ]

    all_records = [
        load_data(file_name % (labeler, business_type)) for labeler in labelers
    ]

    review_id_field = Constants.REVIEW_ID_FIELD
    review_type_field = 'review_type'
    agreement_map = {1: 0, 2: 0, 3: 0}
    reviews_agreement_map = {}
    reviews_labels_map = {}
    reviews_final_label_map = {}

    for row in list(zip(*all_records)):
        my_set = set()
        review_id = row[0][review_id_field]
        labels = []
        for element in row:
            label = element[review_type_field]
            # print(label)
            my_set.add(label)
            labels.append(label)

        reviews_agreement_map[review_id] = len(my_set)
        reviews_labels_map[review_id] = labels
        reviews_final_label_map[review_id] = get_most_common_element(labels)
        agreement_map[len(my_set)] += 1

    print(reviews_agreement_map)
    print(reviews_labels_map)
    print(reviews_final_label_map)
    print(agreement_map)

    # Discard the reviews with 3 different labels
    for review_id, count in reviews_agreement_map.items():
        if count == 3:
            reviews_labels_map.pop(review_id)
            reviews_final_label_map.pop(review_id)

    return reviews_final_label_map


def update_labeled_reviews_records():

    reviews_label_map = compare_records()
    agreed_review_ids = set(reviews_label_map.keys())
    classifier_records = \
        ETLUtils.load_json_file(Constants.CLASSIFIED_RECORDS_FILE)
    classifier_review_ids = \
        {record[Constants.REVIEW_ID_FIELD] for record in classifier_records}
    non_agreed_review_ids = classifier_review_ids.difference(agreed_review_ids)

    # for record in classifier_records:
        # print(record)

    print('number of records before: %d' % len(classifier_records))

    print(reviews_label_map)
    print(non_agreed_review_ids)
    review_type_map = {'s': 'yes', 'g': 'no'}

    # We remove from the classifier records the ones who don't have agreed on a
    # label
    classifier_records = ETLUtils.filter_out_records(
        classifier_records, Constants.REVIEW_ID_FIELD, non_agreed_review_ids)

    # Finally we make the update of the labels
    for record in classifier_records:
        review_id = record[Constants.REVIEW_ID_FIELD]
        record[Constants.SPECIFIC] = review_type_map[reviews_label_map[review_id]]
        # print(record)

    print('number of records after: %d' % len(classifier_records))

    # ETLUtils.save_json_file(
    #     Constants.CLASSIFIED_RECORDS_FILE, classifier_records)


def get_most_common_element(my_list):
    return Counter(my_list).most_common(1)[0][0]


def foo():
    my_records = []
    for i in range(10):
        my_records.append({'column1': i})
    print(my_records)
    to_remove = set(range(1, 10, 2))
    print(to_remove)

    new_records = ETLUtils.filter_out_records(my_records, 'column1', to_remove)
    print(new_records)


def count_specific_generic_ratio(records):
    """
    Prints the proportion of specific and generic documents
    """

    specific_count = 0.0
    generic_count = 0.0

    for record in records:
        if record[Constants.SPECIFIC] == 'yes':
            specific_count += 1
        if record[Constants.SPECIFIC] == 'no':
            generic_count += 1

    print('Specific reviews: %f%%' % (
        specific_count / len(records) * 100))
    print('Generic reviews: %f%%' % (
        generic_count / len(records) * 100))
    print('Specific reviews: %d' % specific_count)
    print('Generic reviews: %d' % generic_count)


def toy_cohens_kappa():
    # rater1 = [1, 1, 1, 0]
    # rater2 = [1, 1, 0, 0]
    # rater3 = [0, 1, 1]
    rater1 = ['s', 's', 's', 'g', 'u']
    rater2 = ['s', 's', 'g', 'g', 's']

    taskdata = [[0, str(i), str(rater1[i])] for i in range(0, len(rater1))] + [
        [1, str(i), str(rater2[i])] for i in range(0, len(rater2))] # + [
                   # [2, str(i), str(rater3[i])] for i in range(0, len(rater3))]
    print(taskdata)
    ratingtask = agreement.AnnotationTask(data=taskdata)
    print("kappa " + str(ratingtask.kappa()))
    print("fleiss " + str(ratingtask.multi_kappa()))
    print("alpha " + str(ratingtask.alpha()))
    print("scotts " + str(ratingtask.pi()))

    print("sklearn kappa " + str(cohen_kappa_score(rater1, rater2)))


def cohens_kappa():

    data_folder = '/Users/fpena/UCC/Thesis/datasets/context/manuallyLabeledReviews/'

    business_type = Constants.ITEM_TYPE
    file_name = data_folder + '%s_%s_reviews.json'

    labelers = [
        # 'francisco',
        'diego',
        'mesut',
        'rohit',
    ]

    all_records = [
        load_data(file_name % (labeler, business_type)) for labeler in labelers
    ]

    rater1 = [record['review_type'] for record in all_records[0]]
    rater2 = [record['review_type'] for record in all_records[1]]
    rater3 = [record['review_type'] for record in all_records[2]]

    taskdata = [[0, str(i), str(rater1[i])] for i in range(0, len(rater1))] + [
        [1, str(i), str(rater2[i])] for i in range(0, len(rater2))] + [
                   [2, str(i), str(rater3[i])] for i in range(0, len(rater3))]
    print(taskdata)
    ratingtask = agreement.AnnotationTask(data=taskdata)
    print("Observed agreement " + str(ratingtask.avg_Ao()))
    print("kappa " + str(ratingtask.kappa()))
    print("fleiss " + str(ratingtask.multi_kappa()))
    print("alpha " + str(ratingtask.alpha()))
    print("scotts " + str(ratingtask.pi()))

    print("sklearn kappa " + str(cohen_kappa_score(rater1, rater2)))
    print("sklearn kappa " + str(cohen_kappa_score(rater1, rater3)))
    print("sklearn kappa " + str(cohen_kappa_score(rater2, rater3)))


def main():
    classifier_records = \
        ETLUtils.load_json_file(Constants.CLASSIFIED_RECORDS_FILE)
    # count_specific_generic_ratio(classifier_records)

    # load_data(file_name)
    # compare_records()
    # update_labeled_reviews_records()
    # foo()
    cohens_kappa()


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)