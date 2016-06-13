
# from fastFM.mcmc import FMRegression
from fastFM import mcmc, als
import numpy
from sklearn.preprocessing import OneHotEncoder

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
    {'user_id': 'U33', 'offering_id': 'WinterHotel1', 'rating': 1.0, 'context': [1.018, 0]}
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

        matrix.append([
            users_map[user_id],
            items_map[item_id],
            record['context'][0],
            record['context'][1]
        ])

    y = numpy.array(y)

    return matrix, y


def predict(records):

    new_matrix, new_y = records_to_matrix(records)
    print(new_matrix)
    encoder = OneHotEncoder(categorical_features=[0, 1], sparse=True)
    encoder.fit(new_matrix)

    new_X = encoder.transform(new_matrix[:-1])
    # print(new_X.todense())

    matrix = [
        [1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
        [1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0],
        [0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0],
        [0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0],
        [0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  1.0,  0.0,  1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  0.0,  1.0,  1.0,  0.0],
        [0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  1.0]
    ]

    y = numpy.array([
        214.51929265, 219.74354687, 216.73769174, 221.61547487, 217.71868329,
        221.41925146, 214.98464738, 219.20650995, 219.24532932, 222.71366784
    ])


    # X = sparse.csc_matrix(matrix)
    # print('******')
    # print(X)

    # X_train, X_test, y_train, y_test = train_test_split(new_X, new_y)
    X_train = new_X
    y_train = new_y[:-1]
    X_test = encoder.transform(new_matrix[-1:])
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


predict(reviews_matrix2)

# new_matrix = records_to_matrix(reviews_matrix)[0]
# train_matrix = new_matrix[:-2]
# test_matrix = new_matrix[-2:]
#
# print(train_matrix)
# print(test_matrix)
# encoder = OneHotEncoder()
# encoder.fit(train_matrix + test_matrix)
# new_x = encoder.transform(train_matrix)
# test_x = encoder.transform(test_matrix)
#
# # print(records_to_matrix(reviews_matrix[:-1]))
# print(new_x.todense())
# print('*****')
# print(test_x.todense())
