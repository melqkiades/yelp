from unittest import TestCase
from evaluation.top_n_evaluator import TopNEvaluator

__author__ = 'fpena'


ratings = [
    {'user_id': 'U1', 'business_id': 'I1', 'stars': 5.0, 'text': 'review U1:I1'},
    {'user_id': 'U1', 'business_id': 'I2', 'stars': 4.0, 'text': 'review U1:I2'},
    {'user_id': 'U1', 'business_id': 'I3', 'stars': 3.0, 'text': 'review U1:I3'},
    {'user_id': 'U1', 'business_id': 'I4', 'stars': 2.0, 'text': 'review U1:I4'},
    {'user_id': 'U1', 'business_id': 'I5', 'stars': 1.0, 'text': 'review U1:I5'},
    {'user_id': 'U1', 'business_id': 'I6', 'stars': 5.0, 'text': 'review U1:I6'},
    {'user_id': 'U1', 'business_id': 'I7', 'stars': 4.0, 'text': 'review U1:I7'},
    {'user_id': 'U1', 'business_id': 'I8', 'stars': 3.0, 'text': 'review U1:I8'},
    {'user_id': 'U1', 'business_id': 'I9', 'stars': 2.0, 'text': 'review U1:I9'},
    {'user_id': 'U2', 'business_id': 'I10', 'stars': 1.0, 'text': 'review U2:I10'},
    {'user_id': 'U2', 'business_id': 'I11', 'stars': 5.0, 'text': 'review U2:I11'},
    {'user_id': 'U2', 'business_id': 'I12', 'stars': 5.0, 'text': 'review U2:I12'},
    {'user_id': 'U3', 'business_id': 'I13', 'stars': 2.0, 'text': 'review U3:I13'},
    {'user_id': 'U3', 'business_id': 'I14', 'stars': 5.0, 'text': 'review U3:I14'},
    {'user_id': 'U4', 'business_id': 'I15', 'stars': 5.0, 'text': 'review U4:I15'},
    {'user_id': 'U5', 'business_id': 'I16', 'stars': 4.0, 'text': 'review U5:I16'}
]

test_set = [
    {'user_id': 'U1', 'business_id': 'I6', 'stars': 5.0, 'text': 'review U1:I6'},
    {'user_id': 'U3', 'business_id': 'I36', 'stars': 5.0, 'text': 'review U3:I36'}
]


class TestTopNEvaluator(TestCase):

    # def test_evaluate(self):
    #     my_predictions_file = None
    #     train_records, test_records =\
    #         ETLUtils.split_train_test(my_records, split=0.8, shuffle_data=False)
    #     my_predictions = rmse_calculator.read_targets_from_txt(my_predictions_file)
    #     top_n_evaluator = TopNEvaluator(my_records, test_records)
    #     top_n_evaluator.find_important_records()
    #     top_n_evaluator.evaluate(my_predictions)

    def test_calculate_important_items(self):
        actual_important_items =\
            TopNEvaluator.find_important_records(ratings)

        expected_important_items = [
            {'user_id': 'U1', 'business_id': 'I1', 'stars': 5.0},
            {'user_id': 'U1', 'business_id': 'I6', 'stars': 5.0},
            {'user_id': 'U2', 'business_id': 'I11', 'stars': 5.0},
            {'user_id': 'U2', 'business_id': 'I12', 'stars': 5.0},
            {'user_id': 'U3', 'business_id': 'I14', 'stars': 5.0},
            {'user_id': 'U4', 'business_id': 'I15', 'stars': 5.0}
        ]

        self.assertItemsEqual(expected_important_items, actual_important_items)

    def test_get_irrelevant_items(self):
        top_n_evaluator = TopNEvaluator(ratings, None)
        top_n_evaluator.initialize()

        actual_irrelevant_items = top_n_evaluator.get_irrelevant_items('U1')
        expected_irrelevant_items = [
            'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16'
        ]
        self.assertItemsEqual(expected_irrelevant_items, actual_irrelevant_items)

        actual_irrelevant_items = top_n_evaluator.get_irrelevant_items('U5')
        expected_irrelevant_items = [
            'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',
            'I10', 'I11', 'I12', 'I13', 'I14', 'I15'
        ]
        self.assertItemsEqual(expected_irrelevant_items, actual_irrelevant_items)

        # top_n_evaluator.get_irrelevant_items('U6')
        self.assertRaises(KeyError, top_n_evaluator.get_irrelevant_items, 'U6')

    def test_create_top_n_lists(self):
        rating_list = {
            'I1': 5.0,
            'I2': 3.0,
            'I3': 1.0,
            'I4': 2.0,
            'I5': 4.5,
            'I6': 3.7
        }

        expected_list_1 = ['I1']
        expected_list_2 = ['I1', 'I5']
        expected_list_3 = ['I1', 'I5', 'I6']
        expected_list_4 = ['I1', 'I5', 'I6', 'I2']
        expected_list_5 = ['I1', 'I5', 'I6', 'I2', 'I4']
        expected_list_6 = ['I1', 'I5', 'I6', 'I2', 'I4', 'I3']
        expected_list_7 = ['I1', 'I5', 'I6', 'I2', 'I4', 'I3']

        self.assertSequenceEqual(
            TopNEvaluator.create_top_n_list(rating_list, 1), expected_list_1)
        self.assertSequenceEqual(
            TopNEvaluator.create_top_n_list(rating_list, 2), expected_list_2)
        self.assertSequenceEqual(
            TopNEvaluator.create_top_n_list(rating_list, 3), expected_list_3)
        self.assertSequenceEqual(
            TopNEvaluator.create_top_n_list(rating_list, 4), expected_list_4)
        self.assertSequenceEqual(
            TopNEvaluator.create_top_n_list(rating_list, 5), expected_list_5)
        self.assertSequenceEqual(
            TopNEvaluator.create_top_n_list(rating_list, 6), expected_list_6)
        self.assertSequenceEqual(
            TopNEvaluator.create_top_n_list(rating_list, 7), expected_list_7)

    def test_get_items_to_predict(self):
        top_n_evaluator = TopNEvaluator(ratings, test_set)
        top_n_evaluator.I = 4
        top_n_evaluator.N = 2
        top_n_evaluator.initialize()
        items_to_predict = top_n_evaluator.get_records_to_predict()

        predictions = [0] * len(test_set) * (top_n_evaluator.I + 1)
        top_n_evaluator.evaluate(predictions)

        print(items_to_predict)

        for item in items_to_predict:
            print(item)

    def test_update_num_hits(self):

        top_n_evaluator = TopNEvaluator([], [])
        self.assertEqual(0, top_n_evaluator.num_generic_hits)
        self.assertEqual(0, top_n_evaluator.num_generic_misses)

        top_n_list = ['I1', 'I2', 'I3', 'I4', 'I5']
        item = 'I3'
        top_n_evaluator.update_num_hits(top_n_list, item)
        self.assertEqual(1, top_n_evaluator.num_generic_hits)
        self.assertEqual(0, top_n_evaluator.num_generic_misses)

        top_n_list = ['I1', 'I2', 'I3', 'I4', 'I5']
        item = 'I6'
        top_n_evaluator.update_num_hits(top_n_list, item)
        self.assertEqual(1, top_n_evaluator.num_generic_hits)
        self.assertEqual(1, top_n_evaluator.num_generic_misses)

        top_n_list = ['I1', 'I6', 'I3', 'I4', 'I5']
        item = 'I4'
        top_n_evaluator.update_num_hits(top_n_list, item)
        self.assertEqual(2, top_n_evaluator.num_generic_hits)
        self.assertEqual(1, top_n_evaluator.num_generic_misses)

    def test_calculate_precision(self):
        N = 5
        top_n_evaluator = TopNEvaluator([], [], N)

        top_n_list = ['I1', 'I2', 'I3', 'I4', 'I5']
        item = 'I3'
        top_n_evaluator.update_num_hits(top_n_list, item)

        top_n_list = ['I1', 'I2', 'I3', 'I4', 'I5']
        item = 'I6'
        top_n_evaluator.update_num_hits(top_n_list, item)

        top_n_list = ['I1', 'I6', 'I3', 'I4', 'I5']
        item = 'I4'
        top_n_evaluator.update_num_hits(top_n_list, item)

        self.assertEqual(2.0 / (3 * N), top_n_evaluator.calculate_precision())

    def test_calculate_recall(self):
        N = 5
        top_n_evaluator = TopNEvaluator([], [], N)

        top_n_list = ['I1', 'I2', 'I3', 'I4', 'I5']
        item = 'I3'
        top_n_evaluator.update_num_hits(top_n_list, item)

        top_n_list = ['I1', 'I2', 'I3', 'I4', 'I5']
        item = 'I6'
        top_n_evaluator.update_num_hits(top_n_list, item)

        top_n_list = ['I1', 'I6', 'I3', 'I4', 'I5']
        item = 'I4'
        top_n_evaluator.update_num_hits(top_n_list, item)

        self.assertEqual(2.0 / 3, top_n_evaluator.calculate_recall())