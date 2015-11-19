import filecmp
import os
from etl.libfm_converter import csv_to_libfm
from unittest import TestCase

__author__ = 'fpena'


folder = '/Users/fpena/UCC/Thesis/projects/yelp/source/python/etl/tests/'


class TestLibfmConverter(TestCase):

    def test_csv_to_libfm(self):

        input_file = folder + 'yelp.csv_train_0'
        expected_file = input_file + ".libfm"
        output_file = expected_file + "_test"

        if os.path.isfile(output_file):
            os.remove(output_file)

        delete_columns = []
        csv_to_libfm(
            input_file, output_file, 2, delete_columns=delete_columns,
            delimiter=',', has_header=True)

        self.assertTrue(filecmp.cmp(output_file, expected_file))

        if os.path.isfile(output_file):
            os.remove(output_file)
