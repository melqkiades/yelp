from yelp.phoenix.business_etl import BusinessETL
from datamining.clusterer import Clusterer

__author__ = 'franpena'









data_folder = '../../../../../../datasets/yelp_phoenix_academic_dataset/'
business_file_path = data_folder + 'yelp_academic_dataset_business.json'
my_matrix = BusinessETL.create_category_matrix(business_file_path)
my_sets = BusinessETL.create_category_sets(business_file_path)
print 'Data pre-processing done'

# Clusterer.cluster_and_evaluate_data(my_matrix, 'k-means-scikit')
# Clusterer.cluster_and_evaluate_data(my_matrix, 'k-means-nltk')
# Clusterer.cluster_and_evaluate_data(my_matrix, 'mean-shift')
Clusterer.cluster_and_evaluate_data(my_matrix, 'ward')
Clusterer.cluster_and_evaluate_data(my_matrix, 'dbscan')
# Clusterer.linkage(my_matrix[:3000])
# Clusterer.gaac(my_matrix[:500][:50])


