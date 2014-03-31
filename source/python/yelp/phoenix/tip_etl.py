import json

__author__ = 'franpena'


class TipETL:
    def __init__(self):
        pass

    @staticmethod
    def load_file(file_path):
        records = [json.loads(line) for line in open(file_path)]

        return records


data_folder = '../../../../../../datasets/yelp_phoenix_academic_dataset/'
tip_file_path = data_folder + 'yelp_academic_dataset_tip.json'
my_records = TipETL.load_file(tip_file_path)
print my_records[1]['text']