
# from fastFM.mcmc import FMRegression
import itertools
from fastFM import mcmc, als, sgd
import numpy
from gensim import corpora
from gensim.models import ldamodel
from sklearn.preprocessing import OneHotEncoder

from etl.libfm_converter import load_libfm_model
from etl.libfm_converter import load_test_file
from utils.constants import Constants

reviews_matrix = [
    {'user_id': 'U01', 'business_id': 'I01', 'stars': 5.0},
    {'user_id': 'U01', 'business_id': 'I02', 'stars': 5.0},
    {'user_id': 'U02', 'business_id': 'I01', 'stars': 5.0},
    {'user_id': 'U02', 'business_id': 'I02', 'stars': 5.0},
    {'user_id': 'U03', 'business_id': 'I01', 'stars': 5.0},
    {'user_id': 'U03', 'business_id': 'I02', 'stars': 5.0},
    {'user_id': 'U04', 'business_id': 'I01', 'stars': 5.0},
    {'user_id': 'U04', 'business_id': 'I02', 'stars': 5.0},
    {'user_id': 'U05', 'business_id': 'I01', 'stars': 5.0},
    {'user_id': 'U05', 'business_id': 'I01', 'stars': 5.0},
    {'user_id': 'U06', 'business_id': 'I01', 'stars': 5.0}
]

# context[0] = Solo/Family
# context[1] = Summer/Winter
reviews_matrix2 = [
    {'user_id': 'U01', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context': [1.001, 0]},
    {'user_id': 'U02', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context': [0.002, 0]},
    {'user_id': 'U03', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context': [0.003, 0]},
    {'user_id': 'U04', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context': [0.004, 0]},
    {'user_id': 'U05', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [0.005, 0]},
    {'user_id': 'U06', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context': [0.006, 1]},
    {'user_id': 'U07', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context': [1.007, 1]},
    {'user_id': 'U08', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [1.008, 0]},
    # {'user_id': 'U09', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [1.009, 0]},
    {'user_id': 'U10', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context': [0.010, 0]},
    {'user_id': 'U11', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context': [0.011, 1]},
    {'user_id': 'U12', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context': [1.012, 1]},
    {'user_id': 'U13', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context': [1.013, 0]},
    {'user_id': 'U14', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [0.014, 0]},
    {'user_id': 'U15', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [1.015, 0]},
    {'user_id': 'U16', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [0.016, 1]},
    {'user_id': 'U17', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [1.017, 1]},
    {'user_id': 'U33', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [1.018, 1]}
]

# context[0] = Solo/Family
# context[1] = Summer/Winter
reviews_matrix3 = [
    {'user_id': 'U01', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context': [1.001, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U02', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context': [1.002, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U03', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context': [0.003, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U04', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context': [0.004, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U05', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [0.005, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U06', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context': [0.006, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U07', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context': [1.007, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U08', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [1.008, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U09', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [1.009, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U10', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context': [0.010, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U11', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context': [0.011, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U12', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context': [1.012, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U13', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context': [1.013, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U14', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [0.014, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U15', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [1.015, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U16', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [0.016, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U17', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [1.017, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U18', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context': [1.001, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U19', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context': [1.002, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U20', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context': [0.003, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U21', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context': [0.004, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U22', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [0.005, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U23', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context': [0.006, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U24', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context': [1.007, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U25', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [1.008, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U26', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [1.009, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U27', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context': [0.010, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U28', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context': [0.011, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U29', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context': [1.012, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U30', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context': [1.013, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U31', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [0.014, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U32', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [1.015, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U33', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [0.016, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U34', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context': [1.017, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context': [1.018, 1],
     'context_text': 'solo winter', 'context_onehot': [0, 1, 0, 1]}
]

train_set = [
    {'user_id': 'U01', 'business_id': 'SummerHotel1', 'stars': 5.0,'context_text': 'family summer'},
    {'user_id': 'U02', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U03', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    {'user_id': 'U04', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    {'user_id': 'U01', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    {'user_id': 'U02', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U03', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context_text': 'family winter'},
    {'user_id': 'U04', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U05', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U01', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    {'user_id': 'U02', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U03', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U04', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U01', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    {'user_id': 'U02', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'family summer'},
    {'user_id': 'U03', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U04', 'business_id': 'SuperbHotel1', 'stars': 5.0,'context_text': 'family winter'},
    # {'user_id': 'U05', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context_text': 'family summer'},
    # {'user_id': 'U19', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    # {'user_id': 'U20', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    # {'user_id': 'U21', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    # {'user_id': 'U22', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    # {'user_id': 'U23', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    # {'user_id': 'U24', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context_text': 'family winter'},
    # {'user_id': 'U25', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    # {'user_id': 'U26', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    # {'user_id': 'U27', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    # {'user_id': 'U28', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    # {'user_id': 'U29', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    # {'user_id': 'U30', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    # {'user_id': 'U31', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    # {'user_id': 'U32', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'family summer'},
    # {'user_id': 'U33', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    # {'user_id': 'U34', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'family winter'}
]

test_set = [
    {'user_id': 'U01', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    {'user_id': 'U03', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    # {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    # {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'winter'},
    # {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    # {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    # {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'summer'},
    # {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    # {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    # {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'winter'},
    # {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    # {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    # {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'summer'},
    # {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    # {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    # {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'solo'},
    # {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    # {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    # {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family'},
    # {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    # {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    # {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'solo'},
    # {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    # {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    # {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'family'},
    # {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'summer'},
    # {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'winter'}
]

my_reviews = [
    "summer garbage01",
    "winter garbage02",
    "solo garbage03",
    # "couple garbage04",
    "family garbage05",
    "summer solo garbage06",
    # "summer couple garbage07",
    "summer family garbage08",
    "winter solo garbage09",
    # "winter couple garbage10",
    "winter family garbage11",
    "summer garbage12",
    "winter garbage13",
    "solo garbage14",
    # "couple garbage15",
    "family garbage16",
    "summer solo garbage17",
    # "summer couple garbage18",
    "summer family garbage19",
    "winter solo garbage20",
    # "winter couple garbage21",
    "winter family garbage22"
]

num_topics = 4


def train_test_records_to_matrix(train_records, test_records):

    matrix, y = records_to_matrix(train_records + test_records)
    train_matrix = matrix[:len(train_records)]
    test_matrix = matrix[len(train_records):]
    train_y = y[:len(train_records)]
    test_y = y[len(train_records):]

    encoder = OneHotEncoder()
    encoder.fit(matrix)
    encoded_train_matrix = encoder.transform(train_matrix)
    encoded_test_matrix = encoder.transform(test_matrix)

    return encoded_train_matrix, encoded_test_matrix, train_y, test_y


def records_to_matrix(records, context_rich_topics=None):
    """
    Converts the given records into a numpy matrix, transforming given each
    string value an integer ID

    :param records: a list of dictionaries with the reviews information
    :type context_rich_topics list[(float, float)]
    :param context_rich_topics: a list that indicates which are the
    context-rich topics. The list contains pairs, in which the first position
    indicates the topic ID, and the second position, the ratio of the topic.
    :return: a numpy matrix with all the independent variables (X) and a numpy
    vector with all the dependent variables (y)
    """

    matrix = []
    y = []

    users_map = {}
    items_map = {}
    user_index = 0
    item_index = 0

    for record in records:
        user_id = record['user_id']
        item_id = record['business_id']
        y.append(record['stars'])

        if user_id not in users_map:
            users_map[user_id] = user_index
            user_index += 1

        if item_id not in items_map:
            items_map[item_id] = item_index
            item_index += 1

        row = [users_map[user_id], items_map[item_id]]

        # for topic in range(num_topics):
        #     topic_probability = record['lda_context'][topic]
        #     row.append(topic_probability)

        if Constants.USE_CONTEXT:
            for topic in context_rich_topics:
                # topic_probability = record['lda_context'][topic[0]]
                # row.append(topic_probability)
                # print(record)
                # print(record[Constants.CONTEXT_TOPICS_FIELD])
                topic_key = 'topic' + str(topic[0])
                topic_probability =\
                    record[Constants.CONTEXT_TOPICS_FIELD][topic_key]
                # print('topic_probability', topic_probability)
                row.append(topic_probability)
                # print('topic_probability', topic_probability)

        matrix.append(row)

    y = numpy.array(y)

    return matrix, y


def predict(train_records, test_records):
    """
    Makes a prediction for the testing set based on the topic probability vector
    of each record and the rating. The topic model is built using the training
    set. This function uses the FastFM Factorization Machines Module for Python

    :param train_records: the training set
    :param test_records: the testing set
    :return: a list with the predictions for the testing set
    """

    records = train_records + test_records
    num_factors = 2

    context_rich_topics = [(i, 1) for i in range(num_topics)]
    new_matrix, new_y = records_to_matrix(records, context_rich_topics)
    print(new_matrix)
    encoder = OneHotEncoder(categorical_features=[0, 1], sparse=True)
    encoder.fit(new_matrix)

    new_x = encoder.transform(new_matrix[:len(train_records)])
    # print(new_x.todense())
    # x_train, x_test, y_train, y_test = train_test_split(new_x, new_y)
    x_train = new_x
    y_train = new_y[:len(train_records)]
    x_test = encoder.transform(new_matrix[len(train_records):])
    # mc_regressor = mcmc.FMRegression(rank=0)
    mc_regressor = sgd.FMRegression(rank=num_factors)
    mc_regressor.fit(x_train, y_train)
    y_pred = mc_regressor.predict(x_test)

    w0 = mc_regressor.w0_
    w = mc_regressor.w_
    V = mc_regressor.V_
    V = V.transpose()
    print('w0', w0)
    print('w', w.shape)
    print('w', w)
    print('V', V.shape)
    print('V', V)
    print('********')
    print(x_test.todense())
    print(y_pred)

    manual_predict(x_test, w0, w, V, num_factors)

    # for i, j in itertools.combinations(range(9), 2):
    #     own_prediction = w0 + w[i] + w[j]
    #     print('My own prediction: %f' % own_prediction)

    als_fm = als.FMRegression(
        n_iter=1000, init_stdev=0.1, rank=num_factors, l2_reg_w=0.1, l2_reg_V=0.5)
    als_fm.fit(x_train, y_train)
    y_pred = als_fm.predict(x_test)

    w0 = als_fm.w0_
    w = als_fm.w_
    V = als_fm.V_
    V = V.transpose()
    print(y_pred)
    manual_predict(x_test, w0, w, V, num_factors)

    return y_pred


def manual_predict(x, w0, w, V, num_factors):

    predictions = []

    for pred_index in range(x.shape[0]):
        sum_ = numpy.zeros(num_factors)
        sum_sqr_ = numpy.zeros(num_factors)

        result = 0.0

        result += w0
        for i in range(len(w)):

            result += w[i] * x[pred_index, i]

        for f in range(num_factors):
            sum_[f] = 0.0
            sum_sqr_[f] = 0.0
            for i in range(len(w)):
                d = V[i, f] * x[pred_index, i]
                sum_[f] += d
                sum_sqr_[f] += d * d
            result += 0.5 * (sum_[f] * sum_[f] - sum_sqr_[f])
        # print('My own prediction: %f' % own_prediction)
        print('My own prediction: %f' % result)

        predictions.append(result)

    return predictions


def preprocess_records(train_records, test_records):
    """
    Creates a bag of words and a corpus for each record and creates a dictionary
    based on all the text contained in the records
    """

    records = train_records + test_records

    all_words = []

    for record in records:
        bow = record['context_text'].split()
        record[Constants.BOW_FIELD] = bow
        all_words.append(bow)

    dictionary = corpora.Dictionary(all_words)

    for record in records:
        record[Constants.CORPUS_FIELD] = \
            dictionary.doc2bow(record[Constants.BOW_FIELD])

    return dictionary


def find_lda_context(train_records, test_records):
    """
    Uses the training records to create a topic model and then updates both
    the training and testing records with a vector of probabilities for each
    topic from the recently created topic model
    """

    dictionary = preprocess_records(train_records, test_records)
    corpus = [record[Constants.CORPUS_FIELD] for record in train_records]
    print(corpus)
    topic_model = ldamodel.LdaModel(
        corpus, id2word=dictionary,
        num_topics=num_topics,
        passes=Constants.TOPIC_MODEL_PASSES,
        iterations=Constants.TOPIC_MODEL_ITERATIONS)

    print(corpus)
    for i in range(num_topics):
        print(topic_model.show_topic(i, topn=2))

    records = train_records + test_records

    for record in records:
        document_topics =\
            topic_model.get_document_topics(record[Constants.CORPUS_FIELD])
        lda_context = [document_topic[1] for document_topic in document_topics]
        record['lda_context'] = lda_context

        context_topics = {}
        for i in range(num_topics):
            topic_id = 'topic' + str(i)
            context_topics[topic_id] = document_topics[i][1]

        record[Constants.CONTEXT_TOPICS_FIELD] = context_topics


def predict_with_lda(train_records, test_records):
    """
    Creates a topic model using the training records and based on the
    probability vector for each document on each record, and the rating makes
    a prediction for every element on the testing records set
    """

    numpy.random.seed(0)
    find_lda_context(train_records, test_records)
    predict(train_records, test_records)


def manual_predict_sample():
    num_users = 4148
    num_items = 284
    num_context_topics = 39
    num_variables = num_users + num_items + num_context_topics

    # unique_id = '3721d1bb5e8644e18e2ecd2e2a30585a'
    # unique_id = 'a872576a0c374e12a42205c79122c905'
    unique_id = '8a998bcf78b04d09900700a51fdc68ed'
    prefix = Constants.GENERATED_FOLDER + unique_id + '_' + Constants.ITEM_TYPE

    saved_model_file = prefix + '_trained_model.libfm'
    test_file = prefix + '_test.csv.libfm'

    w0, w, V = load_libfm_model(saved_model_file, num_variables)
    x_test = load_test_file(test_file, num_variables)
    manual_predict(x_test, w0, w, V, Constants.FM_NUM_FACTORS)


# predict(reviews_matrix2[:-1], reviews_matrix2[-1:])
# predict(reviews_matrix3[:-1], reviews_matrix3[-1:])
# test_reviews_lda(my_reviews)
# find_lda_context(reviews_matrix3[:-1], reviews_matrix3[-1:])
# predict_with_lda(train_set, test_set)
# predict(train_set, test_set)
# manual_predict_sample()
