
import csv
from StringIO import StringIO
import numpy
from etl import ETLUtils

__author__ = 'fpena'


def csv_to_libfm2(
        input_file, output_file, target_column, one_hot_columns,
        delete_columns=None, delimiter=',', has_header=False, num_folds=None):

    if delete_columns is None:
        delete_columns = []

    csv_reader = csv.reader(open(input_file, 'r'), delimiter=delimiter)

    if has_header:
        next(csv_reader)
    data_matrix = list(csv_reader)

    # The static columns are that ones that are not going to be deleted, don'
    # belong to the one-hot columns or the target column
    all_columns = range(len(data_matrix[0]))
    static_columns = set(all_columns).difference(
        [target_column], one_hot_columns, delete_columns)

    id_counter = 0
    output_rows = []

    # Do the target column
    for row_index in range(len(data_matrix)):
        output_rows.append(str(data_matrix[row_index][target_column]))

    # Do the static columns
    for column_index in static_columns:
        # column = data_matrix[:, column_index]
        column = get_column(data_matrix, column_index)

        for row_index in range(len(data_matrix)):
            output_rows[row_index] += " " + str(id_counter) + ":" + str(column[row_index])

        id_counter += 1

    vector_map = {}

    # Do the one-hot columns
    for column_index in one_hot_columns:
        # column = data_matrix[:, column_index]
        column = get_column(data_matrix, column_index)

        for row_index in range(len(data_matrix)):
            vector_key = str(column_index) + " : " + str(column[row_index])

            if vector_key not in vector_map:
                vector_map[vector_key] = id_counter
                id_counter += 1

            output_rows[row_index] += " " + str(vector_map[vector_key]) + ":1"

    if num_folds is None:
        with open(output_file, 'w') as write_file:
            for row in output_rows:
                write_file.write("%s\n" % row)
            return

    for fold in range(num_folds):
        split = 1 - (1/float(num_folds))
        start = float(fold) / num_folds
        train_records, test_records =\
            ETLUtils.split_train_test(output_rows, split, False, start)

        with open(output_file + "_train_" + str(fold), 'w') as write_file:
            for row in train_records:
                write_file.write("%s\n" % row)
        with open(output_file + "_test_" + str(fold), 'w') as write_file:
            for row in test_records:
                write_file.write("%s\n" % row)


def csv_to_libfm(
        input_file, output_file, target_column, delete_columns=None,
        delimiter=',', has_header=False):
    """
    Converts a CSV file to the libFM format.

    :type input_file: str
    :param input_file: the path of the CSV file to convert
    :type output_file: str
    :param output_file: the path of the output libFM file
    :type target_column: int
    :param target_column: the index of the column that contains the target.
    In the case of a recommender system, the target is the rating.
    :type delete_columns: list[int]
    :param delete_columns: A list with the columns of the CSV file that are to
    be excluded
    :type delimiter: str
    :param delimiter: the separator used in the CSV file
    :type has_header: bool
    :param has_header: a boolean indicating if the CSV file has a header or not
    """

    id_counter = 0

    with open(input_file, 'rb') as f:

        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            next(reader)

        # target_list = []
        if delete_columns is None:
            delete_columns = []
        delete_columns.append(target_column)

        vector_map = {}
        output_rows = []

        for row in reader:

            # Remove unwanted columns and the target (rating or dependent)
            # variable
            target = row[target_column]
            for index in sorted(delete_columns, reverse=True):
                row.pop(index)

            output_row = str(target)

            for i in range(len(row)):
                vector_key = str(i) + " : " + row[i]

                if vector_key not in vector_map:
                    vector_map[vector_key] = id_counter
                    id_counter += 1

                output_row += " " + str(vector_map[vector_key]) + ":1"
            output_rows.append(output_row)

    with open(output_file, 'w') as write_file:
        for row in output_rows:
            write_file.write("%s\n" % row)


def get_column(matrix, i):
    return [row[i] for row in matrix]


def main():

    my_input_folder = '/Users/fpena/tmp/libfm-1.42.src/scripts/'
    my_input_folder = '/Users/fpena/UCC/Thesis/datasets/context/'
    json_file = my_input_folder + 'yelp_training_set_review_hotels.json'
    # my_output_file = my_input_file + ".libfm"
    # my_input_file = my_input_folder + 'u1.base'
    # my_records = ETLUtils.load_json_file(my_input_file)
    my_delete_columns = []

    # csv_to_libfm(
    #     json_file, my_output_file, 2, delete_columns=my_delete_columns,
    #     delimiter=',', has_header=True)

    my_export_folder = '/Users/fpena/tmp/libfm-1.42.src/scripts/'
    # my_export_file = my_export_folder + 'yelp_training_set_review_hotels_shuffled.csv'
    csv_file = my_export_folder + 'yelp3.csv'
    libfm_file = my_export_folder + 'yelp_delete.libfm'
    # ETLUtils.json_to_csv(json_file, csv_file, 'user_id', 'business_id', 'stars')
    # csv_to_libfm(
    #     csv_file, libfm_file, 2, delete_columns=my_delete_columns,
    #     delimiter=',', has_header=True)


    # ETLUtils.json_to_csv(json_file, csv_file, 'user_id', 'business_id', 'stars', False, True)


    # csv_file = my_export_folder + 'yelp3.csv'
    # libfm_file = my_export_folder + 'yelp_delete.libfm'
    # csv_to_libfm(
    #     csv_file, libfm_file, 2, delete_columns=None,
    #     delimiter=',', has_header=True)
    #
    # csv_file = my_export_folder + 'yelp2.csv'
    # libfm_file = my_export_folder + 'yelp_delete2.libfm'
    # csv_to_libfm(
    #     csv_file, libfm_file, 2, delete_columns=None,
    #     delimiter=',', has_header=True)

    csv_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_hotel_context_shuffled.csv"
    libfm_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_hotel_context_shuffled.libfm"
    csv_to_libfm2(csv_file, libfm_file, 0, [1, 2], [], ',', has_header=True, num_folds=5)

main()

# d = StringIO("0.4,21,72,33\n0.1,35,58,44\n0.9,18,71,33\n0.9,18,71,44")
# csv_file = '/Users/fpena/tmp/libfm-1.42.src/scripts/sample.csv'
# libfm_file = '/Users/fpena/tmp/libfm-1.42.src/scripts/sample.libfm'
# csv_to_libfm2(csv_file, libfm_file, 1, [4], [2])