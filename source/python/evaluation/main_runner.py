import argparse

import time

from evaluation import context_top_n_runner
from utils.constants import Constants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--fold', metavar='int', type=int,
        nargs=1, help='The index of the cross validation fold')

    args = parser.parse_args()
    fold = args.fold[0]

    new_properties = {
        Constants.NESTED_CROSS_VALIDATION_CYCLE_FIELD: fold,
        Constants.CROSS_VALIDATION_STRATEGY_FIELD: 'nested_validate'
    }

    Constants.update_properties(new_properties)

    context_top_n_runner.run_tests()

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print("Total time = %f seconds" % total_time)
