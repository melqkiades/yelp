import operator
import time
from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


def calculate_top_n_precision(reviews, recommender, n, min_score, num_folds):

    start_time = time.time()
    split = 1 - (1/float(num_folds))
    total_precision = 0.
    num_cycles = 0

    for i in xrange(0, num_folds):
        start = float(i) / num_folds
        train, test = ETLUtils.split_train_test(reviews, split=split, shuffle_data=False, start=start)
        recommender.load(train)
        user_ids = recommender.user_ids

        for user_id in user_ids:
            precision = calculate_recommender_precision(test, user_id, recommender, n, min_score)

            if precision is not None:
                total_precision += precision
                num_cycles += 1

    final_precision = total_precision / num_cycles
    execution_time = time.time() - start_time

    print('Final Top N Precision: %f' % final_precision)
    print("--- %s seconds ---" % execution_time)

    return final_precision


def calculate_recommender_precision(test_data, user_id, recommender, n, min_score):

    known_ratings = extractor.get_user_item_ratings(test_data, user_id, True)
    items = known_ratings.keys()

    predicted_ratings = {}

    for item in items:
        predicted_ratings[item] = recommender.predict_rating(user_id, item)

    return calculate_precision(known_ratings, predicted_ratings, n, min_score)


def calculate_precision(known_ratings, predicted_ratings, n, min_score):

    if not known_ratings:
        return None

    sorted_predicted_ratings =\
        sorted(predicted_ratings.iteritems(), key=operator.itemgetter(1), reverse=True)[:n]

    num_hits = 0.
    for item, predicted_rating in sorted_predicted_ratings:
        known_rating = known_ratings[item]

        # print(item, known_rating)

        if known_rating >= min_score:
            num_hits += 1

    # print(num_hits)

    return num_hits / min(n, len(known_ratings))
