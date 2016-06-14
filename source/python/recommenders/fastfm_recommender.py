
# from fastFM.mcmc import FMRegression
from fastFM import mcmc, als
import numpy
from gensim import corpora
from gensim.models import ldamodel
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder

from utils.constants import Constants

reviews_matrix = [
    {'user_id': 'U01', 'offering_id': 'I01', 'rating': 5.0},
    {'user_id': 'U01', 'offering_id': 'I02', 'rating': 5.0},
    {'user_id': 'U02', 'offering_id': 'I01', 'rating': 5.0},
    {'user_id': 'U02', 'offering_id': 'I02', 'rating': 5.0},
    {'user_id': 'U03', 'offering_id': 'I01', 'rating': 5.0},
    {'user_id': 'U03', 'offering_id': 'I02', 'rating': 5.0},
    {'user_id': 'U04', 'offering_id': 'I01', 'rating': 5.0},
    {'user_id': 'U04', 'offering_id': 'I02', 'rating': 5.0},
    {'user_id': 'U05', 'offering_id': 'I01', 'rating': 5.0},
    {'user_id': 'U05', 'offering_id': 'I01', 'rating': 5.0},
    {'user_id': 'U06', 'offering_id': 'I01', 'rating': 5.0}
]

# context[0] = Solo/Family
# context[1] = Summer/Winter
reviews_matrix2 = [
    {'user_id': 'U01', 'offering_id': 'SummerHotel1', 'rating': 5.0, 'context': [1.001, 0]},
    {'user_id': 'U02', 'offering_id': 'SummerHotel1', 'rating': 5.0, 'context': [0.002, 0]},
    {'user_id': 'U03', 'offering_id': 'SummerHotel1', 'rating': 5.0, 'context': [0.003, 0]},
    {'user_id': 'U04', 'offering_id': 'SummerHotel1', 'rating': 5.0, 'context': [0.004, 0]},
    {'user_id': 'U05', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [0.005, 0]},
    {'user_id': 'U06', 'offering_id': 'WinterHotel1', 'rating': 5.0, 'context': [0.006, 1]},
    {'user_id': 'U07', 'offering_id': 'WinterHotel1', 'rating': 5.0, 'context': [1.007, 1]},
    {'user_id': 'U08', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [1.008, 0]},
    # {'user_id': 'U09', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [1.009, 0]},
    {'user_id': 'U10', 'offering_id': 'BusinessHotel1', 'rating': 5.0, 'context': [0.010, 0]},
    {'user_id': 'U11', 'offering_id': 'BusinessHotel1', 'rating': 5.0, 'context': [0.011, 1]},
    {'user_id': 'U12', 'offering_id': 'BusinessHotel1', 'rating': 1.0, 'context': [1.012, 1]},
    {'user_id': 'U13', 'offering_id': 'BusinessHotel1', 'rating': 1.0, 'context': [1.013, 0]},
    {'user_id': 'U14', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [0.014, 0]},
    {'user_id': 'U15', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [1.015, 0]},
    {'user_id': 'U16', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [0.016, 1]},
    {'user_id': 'U17', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [1.017, 1]},
    {'user_id': 'U33', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [1.018, 1]}
]

# context[0] = Solo/Family
# context[1] = Summer/Winter
reviews_matrix3 = [
    {'user_id': 'U01', 'offering_id': 'SummerHotel1', 'rating': 5.0, 'context': [1.001, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U02', 'offering_id': 'SummerHotel1', 'rating': 1.0, 'context': [1.002, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U03', 'offering_id': 'SummerHotel1', 'rating': 5.0, 'context': [0.003, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U04', 'offering_id': 'SummerHotel1', 'rating': 1.0, 'context': [0.004, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U05', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [0.005, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U06', 'offering_id': 'WinterHotel1', 'rating': 5.0, 'context': [0.006, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U07', 'offering_id': 'WinterHotel1', 'rating': 5.0, 'context': [1.007, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U08', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [1.008, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U09', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [1.009, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U10', 'offering_id': 'BusinessHotel1', 'rating': 5.0, 'context': [0.010, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U11', 'offering_id': 'BusinessHotel1', 'rating': 5.0, 'context': [0.011, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U12', 'offering_id': 'BusinessHotel1', 'rating': 1.0, 'context': [1.012, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U13', 'offering_id': 'BusinessHotel1', 'rating': 1.0, 'context': [1.013, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U14', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [0.014, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U15', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [1.015, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U16', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [0.016, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U17', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [1.017, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U18', 'offering_id': 'SummerHotel1', 'rating': 5.0, 'context': [1.001, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U19', 'offering_id': 'SummerHotel1', 'rating': 1.0, 'context': [1.002, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U20', 'offering_id': 'SummerHotel1', 'rating': 5.0, 'context': [0.003, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U21', 'offering_id': 'SummerHotel1', 'rating': 1.0, 'context': [0.004, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U22', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [0.005, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U23', 'offering_id': 'WinterHotel1', 'rating': 5.0, 'context': [0.006, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U24', 'offering_id': 'WinterHotel1', 'rating': 5.0, 'context': [1.007, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U25', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [1.008, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U26', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [1.009, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U27', 'offering_id': 'BusinessHotel1', 'rating': 5.0, 'context': [0.010, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U28', 'offering_id': 'BusinessHotel1', 'rating': 5.0, 'context': [0.011, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U29', 'offering_id': 'BusinessHotel1', 'rating': 1.0, 'context': [1.012, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U30', 'offering_id': 'BusinessHotel1', 'rating': 1.0, 'context': [1.013, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U31', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [0.014, 0],
     'context_text': 'solo summer', 'context_onehot': [1, 0, 1, 0]},
    {'user_id': 'U32', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [1.015, 0],
     'context_text': 'family summer', 'context_onehot': [0, 1, 1, 0]},
    {'user_id': 'U33', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [0.016, 1],
     'context_text': 'solo winter', 'context_onehot': [1, 0, 0, 1]},
    {'user_id': 'U34', 'offering_id': 'SuperbHotel1', 'rating': 5.0, 'context': [1.017, 1],
     'context_text': 'family winter', 'context_onehot': [0, 1, 0, 1]},
    {'user_id': 'U83', 'offering_id': 'SuperbHotel1', 'rating': 1.0, 'context': [1.018, 1],
     'context_text': 'summer', 'context_onehot': [0, 1, 0, 1]}
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


def records_to_matrix(records):

    matrix = []
    y = []

    users_map = {}
    items_map = {}
    user_index = 0
    item_index = 0

    for record in records:
        user_id = record['user_id']
        item_id = record['offering_id']
        y.append(record['rating'])

        if user_id not in users_map:
            users_map[user_id] = user_index
            user_index += 1

        if item_id not in items_map:
            items_map[item_id] = item_index
            item_index += 1

        # matrix.append([
        #     users_map[user_id],
        #     items_map[item_id],
        #     record['context'][0],
        #     record['context'][1]
        # ])
        matrix.append([
            users_map[user_id],
            items_map[item_id],
            # record['context_onehot'][0],
            # record['context_onehot'][1],
            # record['context_onehot'][2],
            # record['context_onehot'][3],
            record['lda_context'][0],
            record['lda_context'][1],
            record['lda_context'][2],
            record['lda_context'][3]
        ])

    y = numpy.array(y)

    return matrix, y


def predict(train_records, test_records):

    records = train_records + test_records

    new_matrix, new_y = records_to_matrix(records)[:len(train_records)]
    print(new_matrix)
    encoder = OneHotEncoder(categorical_features=[0, 1], sparse=True)
    encoder.fit(new_matrix)

    new_X = encoder.transform(new_matrix[:len(train_records)])
    # print(new_X.todense())
    # X_train, X_test, y_train, y_test = train_test_split(new_X, new_y)
    X_train = new_X
    y_train = new_y[:len(train_records)]
    X_test = encoder.transform(new_matrix[len(train_records):])
    mc_regressor = mcmc.FMRegression()
    y_pred = mc_regressor.fit_predict(X_train, y_train, X_test)
    print('********')
    print(X_test.todense())
    print(y_pred)

    als_fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1,
                          l2_reg_V=0.5)
    als_fm.fit(X_train, y_train)
    y_pred = als_fm.predict(X_test)
    print(y_pred)

    return y_pred


def create_bag_of_words(document_list):
    """
    Creates a bag of words representation of the document list given. It removes
    the punctuation and the stop words.

    :type document_list: list[str]
    :param document_list:
    :rtype: list[list[str]]
    :return:
    """
    tokenizer = RegexpTokenizer(r'\w+')
    cached_stop_words = set(stopwords.words("english"))
    body = []
    processed = []

    # remove common words and tokenize
    # texts = [[word for word in document.lower().split() if word not in stopwords.words('english')]
    #          for document in reviews]

    for i in range(0, len(document_list)):
        body.append(document_list[i].lower())

    for entry in body:
        row = tokenizer.tokenize(entry)
        processed.append([word for word in row if word not in cached_stop_words])

    return processed


def train_lda(reviews):
    numpy.random.seed(0)

    num_topics = 4
    bag_of_words = create_bag_of_words(reviews)
    dictionary = corpora.Dictionary(bag_of_words)
    corpus = [dictionary.doc2bow(text) for text in bag_of_words]
    topic_model = ldamodel.LdaModel(
        corpus, id2word=dictionary,
        num_topics=num_topics,
        passes=Constants.LDA_MODEL_PASSES,
        iterations=Constants.LDA_MODEL_ITERATIONS)

    print(bag_of_words)
    print(corpus)
    for i in range(num_topics):
        print(topic_model.show_topic(i, topn=2))

    return topic_model, dictionary



def test_reviews_lda(reviews):

    topic_model, dictionary = train_lda(reviews)

    print(reviews)


    # query = dictionary.doc2bow(["black", "sabbath"])
    # query = dictionary.doc2bow(["deep", "purple", "acdc", "led", "fdsfre"])
    query = dictionary.doc2bow(["summer", "solo"])
    print(query)

    text = "solo"
    bow = create_bag_of_words([text])
    print(bow)
    query = dictionary.doc2bow(bow[0])
    print(query)
    print(topic_model.get_document_topics(query))


def preprocess_records(train_records, test_records):
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

    num_topics = 4
    dictionary = preprocess_records(train_records, test_records)
    corpus = [record[Constants.CORPUS_FIELD] for record in train_records]
    print(corpus)
    topic_model = ldamodel.LdaModel(
        corpus, id2word=dictionary,
        num_topics=num_topics,
        passes=Constants.LDA_MODEL_PASSES,
        iterations=Constants.LDA_MODEL_ITERATIONS)

    # print(bag_of_words)
    print(corpus)
    for i in range(num_topics):
        print(topic_model.show_topic(i, topn=2))

    records = train_records + test_records

    for record in records:
        document_topics = topic_model.get_document_topics(record[Constants.CORPUS_FIELD])
        lda_context = [document_topic[1] for document_topic in document_topics]
        record['lda_context'] = lda_context
        # print(document_topics)
        # print(lda_context)


def predict_with_lda(train_records, test_records):

    find_lda_context(train_records, test_records)
    predict(train_records, test_records)



# predict(reviews_matrix2[:-1], reviews_matrix2[-1:])
# predict(reviews_matrix3[:-1], reviews_matrix3[-1:])
# test_reviews_lda(my_reviews)
# find_lda_context(reviews_matrix3[:-1], reviews_matrix3[-1:])
predict_with_lda(reviews_matrix3[:-1], reviews_matrix3[-1:])
