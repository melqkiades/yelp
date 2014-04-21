import numpy
from etl import ETLUtils
from yelp.phoenix.business_etl import BusinessETL
from datamining.clusterer import Clusterer

__author__ = 'franpena'



def get_categories(file_path):
    records = ETLUtils.load_json_file(file_path)

    # Now we obtain the categories for all the businesses
    records = ETLUtils.add_transpose_list_column('categories', records)
    BusinessETL.drop_unwanted_fields(records)

    return records[0].keys()


def binary_to_categories(binary_list, categories):

    num_categories = len(categories)

    category_list = []

    for i in xrange(num_categories):
        if binary_list[i] == 1:
            category_list.append(categories[i])

    # print category_list

    return category_list






data_folder = '../../../../../../datasets/yelp_phoenix_academic_dataset/'
business_file_path = data_folder + 'yelp_academic_dataset_business.json'
my_matrix = BusinessETL.create_category_matrix(business_file_path)
my_sets = BusinessETL.create_category_sets(business_file_path)
print 'Data pre-processing done'

# Clusterer.cluster_and_evaluate_data(my_matrix, 'k-means-scikit')
# Clusterer.cluster_and_evaluate_data(my_matrix, 'k-means-nltk')
# Clusterer.cluster_and_evaluate_data(my_matrix, 'mean-shift')
# Clusterer.cluster_and_evaluate_data(my_matrix, 'ward')
# Clusterer.cluster_and_evaluate_data(my_matrix, 'dbscan')
my_labels = Clusterer.cluster_data(my_matrix, 'dbscan')
my_categories = get_categories(business_file_path)

size = len(set(my_labels))
clusters = [[] for i in range(size)]

for i in xrange(len(my_labels)):
    if my_labels[i] == -1:
        clusters[size-1].append(binary_to_categories(my_matrix[i], my_categories))
    else:
        clusters[int(my_labels[i])].append(binary_to_categories(my_matrix[i], my_categories))
    # print my_labels[i]
# Clusterer.linkage(my_matrix[:3000])
# Clusterer.gaac(my_matrix[:500][:50])

sets = []
for cluster in clusters:
    my_set = set()
    for my_list in cluster:
        for element in my_list:
            my_set.add(element)
    print(my_set)
    sets.append(list(my_set))

print 'Size', size

import csv
with open('clusters.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(clusters)

with open('sets.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(sets)
