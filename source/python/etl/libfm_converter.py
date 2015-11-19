
import csv

__author__ = 'fpena'


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


def main():

    my_input_folder = '/Users/fpena/tmp/libfm-1.42.src/scripts/'
    my_input_file = my_input_folder + 'yelp.csv_train_0'
    my_output_file = my_input_file + ".libfm"
    # my_input_file = my_input_folder + 'u1.base'
    # my_records = ETLUtils.load_json_file(my_input_file)
    my_delete_columns = []

    csv_to_libfm(
        my_input_file, my_output_file, 2, delete_columns=my_delete_columns,
        delimiter=',', has_header=True)

# main()
