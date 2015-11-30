
import csv

__author__ = 'fpena'


def csv_to_libfm(
        input_files, target_column, one_hot_columns,
        delete_columns=None, delimiter=',', has_header=False):
    """
    Converts a CSV file to the libFM format.

    :type input_files: list[str]
    :param input_files: a list with the path of the CSV files to convert
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

    if delete_columns is None:
        delete_columns = []

    csv_reader_list = [
        csv.reader(open(input_file, 'r'), delimiter=delimiter)
        for input_file in input_files
        ]

    data_matrix_list = []

    for csv_reader in csv_reader_list:
        if has_header:
            next(csv_reader)
        data_matrix_list.append(list(csv_reader))

    # The static columns are that ones that are not going to be deleted, don't
    # belong to the one-hot columns or the target column
    all_columns = range(len(data_matrix_list[0][0]))
    static_columns = set(all_columns).difference(
        [target_column], one_hot_columns, delete_columns)

    id_counter = 0
    output_row_list = []

    # Do the target column
    for data_matrix in data_matrix_list:
        output_rows = []
        for row_index in range(len(data_matrix)):
            output_rows.append(str(data_matrix[row_index][target_column]))
        output_row_list.append(output_rows)

    # Do the static columns
    for column_index in static_columns:

        for data_matrix, output_rows in zip(data_matrix_list, output_row_list):
            column = get_column(data_matrix, column_index)

            for row_index in range(len(data_matrix)):
                output_rows[row_index] +=\
                    " " + str(id_counter) + ":" + str(column[row_index])

        id_counter += 1

    vector_map = {}

    # Do the one-hot columns
    for data_matrix, output_rows in zip(data_matrix_list, output_row_list):
        for column_index in one_hot_columns:
            column = get_column(data_matrix, column_index)

            for row_index in range(len(data_matrix)):
                vector_key = str(column_index) + " : " + str(column[row_index])

                if vector_key not in vector_map:
                    vector_map[vector_key] = id_counter
                    id_counter += 1

                output_rows[row_index] += " " + str(vector_map[vector_key]) + ":1"

    for input_file, output_rows in zip(input_files, output_row_list):
        with open(input_file + '.libfm', 'w') as write_file:
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

    csv_files = [
        "/Users/fpena/UCC/Thesis/datasets/context/yelp_hotel_context_shuffled.csv"
    ]
    libfm_file = "/Users/fpena/UCC/Thesis/datasets/context/yelp_hotel_context_shuffled.libfm"
    csv_to_libfm(csv_files, 0, [1, 2], [], ',', has_header=True)

# main()

# d = StringIO("0.4,21,72,33\n0.1,35,58,44\n0.9,18,71,33\n0.9,18,71,44")
csv_files = [
    '/Users/fpena/tmp/libfm-1.42.src/scripts/sample.csv',
    '/Users/fpena/tmp/libfm-1.42.src/scripts/sample2.csv'
]
# libfm_file = '/Users/fpena/tmp/libfm-1.42.src/scripts/sample.libfm'
csv_to_libfm(csv_files, 1, [4], [2])
