from etl import ETLUtils

__author__ = 'fpena'


# records = ETLUtils.load_csv_file('/Users/fpena/UCC/Thesis/datasets/ml-1m.csv', '|')
# records = ETLUtils.load_csv_file('/Users/fpena/UCC/Thesis/datasets/ml-10m.csv', '|')

def get_ml_100K_dataset():
    records = ETLUtils.load_csv_file('/Users/fpena/UCC/Thesis/datasets/ml-100k.csv', '\t')
    for record in records:
        record['overall_rating'] = float(record['overall_rating'])
    return records

# print(records[0])
# print(records[1])
# print(records[10])
# print(records[100])
