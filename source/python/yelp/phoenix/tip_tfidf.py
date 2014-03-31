from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from etl.etl_utils import ETLUtils
from yelp.phoenix.tip_etl import TipETL
import numpy as np
import nltk

__author__ = 'franpena'


class TipTfidf:
    def __init__(self):
        pass

    @staticmethod
    def tf_idf(file_path):
        records = TipETL.load_file(file_path)
        data = [record['text'] for record in records]
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        train = vectorizer.fit_transform(data)
        #print "Vocabulary:", vectorizer.get_feature_names()
        num_samples, num_features = train.shape
        print("#samples: %d, #features: %d" % (
            num_samples, num_features))

        business_records = ETLUtils.filter_records(records, 'business_id', ['uFJwKlHL6HyHSJmORO8-5w'])
        business_data = [record['text'] for record in business_records]
        freq_term_matrix = vectorizer.transform(business_data)
        vocabulary = vectorizer.get_feature_names()

        my_list = []
        rows, cols = freq_term_matrix.nonzero()
        for row, col in zip(rows, cols):
            my_dict = {}
            word = vocabulary[col]
            my_dict['tip_id'] = row
            my_dict['word'] = word
            my_dict['tfidf'] = freq_term_matrix[row, col]
            my_list.append(my_dict)

        data_frame = DataFrame(my_list)
        suma = data_frame.groupby('word').aggregate(np.sum)['tfidf']
        ordenado = suma.order()
        print ordenado

        #for row in freq_term_matrix:
            #print(row)

        #Stemmer
        stemmer = nltk.stem.SnowballStemmer('english')
        #print(stemmer.stem("graphics"))

    @staticmethod
    def analyze(file_path):
        records = TipETL.load_file(file_path)
        ETLUtils.drop_fields(['text', 'type', 'date', 'user_id', 'likes'],
                             records)
        data_frame = DataFrame(records)
        counts = data_frame.groupby('business_id').size()
        counts.sort(ascending=0)
        top_counts = counts[:1000]
        print(top_counts)

        print records[0].keys()

#samples: 2, #features: 37


data_folder = '../../../../../../datasets/yelp_phoenix_academic_dataset/'
tip_file_path = data_folder + 'yelp_academic_dataset_tip.json'
my_records = TipETL.load_file(tip_file_path)
TipTfidf.tf_idf(tip_file_path)
#TipTfidf.analyze(tip_file_path)

