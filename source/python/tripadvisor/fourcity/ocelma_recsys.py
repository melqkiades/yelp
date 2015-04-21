from etl import ETLUtils
from evaluation.mean_absolute_error import MeanAbsoluteError
from evaluation.root_mean_square_error import RootMeanSquareError

__author__ = 'fpena'


from recsys.algorithm.factorize import SVD
svd = SVD()
# file_name = '/Users/fpena/UCC/Thesis/datasets/ml-1m/ratings.dat'
# svd.load_data(filename=file_name, sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})
file_name = '/Users/fpena/tmp/reviews.csv'
file_name_header = '/Users/fpena/tmp/reviews-header.csv'
# file_name = '/Users/fpena/tmp/small-reviews-matrix.csv'
# file_name_header = '/Users/fpena/tmp/small-reviews-header.csv'
svd.load_data(filename=file_name, sep='|', format={'col':0, 'row':1, 'value':2, 'ids': str})

k = 100
svd.compute(k=k, min_values=10, pre_normalize=None, mean_center=True, post_normalize=True)
# predicted_rating = svd.predict(int(5), 'A1', 1, 10)
# predicted_rating2 = svd.predict(int(1), 'A1', 1, 10)

# print('Predicted rating', predicted_rating)
# print('Predicted rating', predicted_rating2)

records = ETLUtils.load_csv_file(file_name_header, '|')
errors = []

for record in records:
    try:
        # print(record['user'], record['item'], record['rating'])
        user = record['user']
        item = int(record['item'])
        predicted_rating = svd.predict(item, user, 1, 5)
        print(record['user'], record['item'], predicted_rating)
        # predicted_rating = round(predicted_rating)
        actual_rating = svd.get_matrix().value(item, user)
        error = abs(predicted_rating - actual_rating)
        errors.append(error)
    except KeyError:
        continue

mean_absolute_error = MeanAbsoluteError.compute_list(errors)
root_mean_square_error = RootMeanSquareError.compute_list(errors)
print('Mean Absolute error: %f' % mean_absolute_error)
print('Root mean square error: %f' % root_mean_square_error)
