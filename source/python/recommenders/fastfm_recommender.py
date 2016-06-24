
# from fastFM.mcmc import FMRegression
from fastFM import mcmc, als
import numpy
from gensim import corpora
from gensim.models import ldamodel
from sklearn.preprocessing import OneHotEncoder

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
    {'user_id': 'U05', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    {'user_id': 'U06', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U07', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context_text': 'family winter'},
    {'user_id': 'U08', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U09', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U10', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    {'user_id': 'U11', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U12', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U13', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U14', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    {'user_id': 'U15', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'family summer'},
    {'user_id': 'U16', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U17', 'business_id': 'SuperbHotel1', 'stars': 5.0,'context_text': 'family winter'},
    {'user_id': 'U18', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context_text': 'family summer'},
    {'user_id': 'U19', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U20', 'business_id': 'SummerHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    {'user_id': 'U21', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    {'user_id': 'U22', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    {'user_id': 'U23', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U24', 'business_id': 'WinterHotel1', 'stars': 5.0, 'context_text': 'family winter'},
    {'user_id': 'U25', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U26', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U27', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    {'user_id': 'U28', 'business_id': 'BusinessHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U29', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U30', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U31', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'solo summer'},
    {'user_id': 'U32', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'family summer'},
    {'user_id': 'U33', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'solo winter'},
    {'user_id': 'U34', 'business_id': 'SuperbHotel1', 'stars': 5.0, 'context_text': 'family winter'}
]

test_set = [
    {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'winter'},
    {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U83', 'business_id': 'WinterHotel1', 'stars': 1.0, 'context_text': 'summer'},
    {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'winter'},
    {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U83', 'business_id': 'SummerHotel1', 'stars': 1.0, 'context_text': 'summer'},
    {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'solo'},
    {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U83', 'business_id': 'BusinessHotel1', 'stars': 1.0, 'context_text': 'family'},
    {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'solo winter'},
    {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'solo summer'},
    {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'solo'},
    {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'family winter'},
    {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'family summer'},
    {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'family'},
    {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'summer'},
    {'user_id': 'U83', 'business_id': 'SuperbHotel1', 'stars': 1.0, 'context_text': 'winter'}
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
    mc_regressor = mcmc.FMRegression()
    y_pred = mc_regressor.fit_predict(x_train, y_train, x_test)
    print('********')
    print(x_test.todense())
    print(y_pred)

    als_fm = als.FMRegression(
        n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
    als_fm.fit(x_train, y_train)
    y_pred = als_fm.predict(x_test)
    print(y_pred)

    return y_pred


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
        passes=Constants.LDA_MODEL_PASSES,
        iterations=Constants.LDA_MODEL_ITERATIONS)

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


# predict(reviews_matrix2[:-1], reviews_matrix2[-1:])
# predict(reviews_matrix3[:-1], reviews_matrix3[-1:])
# test_reviews_lda(my_reviews)
# find_lda_context(reviews_matrix3[:-1], reviews_matrix3[-1:])
# predict_with_lda(train_set, test_set)
