from collections import defaultdict
import csv
from etl import ETLUtils

__author__ = 'fpena'


def load_csv_file(file_path):

    records = []
    rating_criteria = [
        'cleanliness_rating',
        'location_rating',
        'rooms_rating',
        'service_rating',
        # 'sleep_quality_rating',
        'value_rating',
        'overall_rating'
    ]

    with open(file_path) as read_file:
        reader = csv.DictReader(read_file)  # read rows into a dictionary format
        for row in reader:
            dictionary = {}
            for (key, value) in row.items(): # go over each column name and value
                if key in rating_criteria:
                    dictionary[key] = float(value)
                else:
                    dictionary[key] = value
            records.append(dictionary)

    return records


# file_dir = '/Users/fpena/UCC/Thesis/datasets/TripAdvisor/PrefLib/trip/'
# file_name = 'CD-00001-00000001.dat'
#
#
# my_records = ETLUtils.load_csv_file(file_dir + file_name, ',')
# print(len(my_records))
