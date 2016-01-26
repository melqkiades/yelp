import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from evaluation.root_mean_square_error import RootMeanSquareError

__author__ = 'fpena'


def calculate_rmse_from_csv(
        ratings_file, predictions_file, target_column, has_header=True,
        delimiter=','):

    pass


def read_targets_from_txt(text_file):

    target_list = []
    with open(text_file) as f:
        for line in f:
            target_list.append(float(line))

    return target_list


def read_targets_from_csv(
        csv_file, target_column, has_header=True, delimiter=','):

    target_list = []

    with open(csv_file, 'rb') as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            next(reader)

        for row in reader:
            target_list.append(float(row[target_column]))

    return target_list


def calculate_mae(true_values, predictions):
    return mean_absolute_error(true_values, predictions)


def calculate_rmse(true_values, predictions):
    return mean_squared_error(true_values, predictions) ** 0.5


def calculate_rmse2(true_values, predictions):
    rmse = RootMeanSquareError()
    for true_value, prediction in zip(true_values, predictions):
        rmse.add(true_value, prediction)

    return rmse.compute()


def main():

    my_file = '/Users/fpena/tmp/libfm-1.42.src/scripts/yelp.csv_test_0'
    my_text_file = '/Users/fpena/tmp/libfm-1.42.src/bin/results.txt'
    true_values = read_targets_from_csv(my_file, 2)
    predictions = read_targets_from_txt(my_text_file)
    print(true_values)
    print(predictions)
    print(calculate_rmse(true_values, predictions))
    print(calculate_rmse2(true_values, predictions))

# main()

