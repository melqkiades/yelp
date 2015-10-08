import operator
from random import shuffle
import time
from etl import ETLUtils
from topicmodeling.context import reviews_clusterer
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


def calculate_top_n_precision(reviews, recommender, n, min_score, num_folds):

    start_time = time.time()
    split = 1 - (1/float(num_folds))
    total_precision = 0.
    num_cycles = 0

    for i in xrange(0, num_folds):
        print('Fold', i )
        start = float(i) / num_folds
        train, test = ETLUtils.split_train_test(
            reviews, split=split, shuffle_data=False, start=start)
        recommender.load(train)
        user_ids = recommender.user_ids

        for user_id in user_ids:
            precision = calculate_recommender_precision(
                test, user_id, recommender, n, min_score)

            if precision is not None:
                total_precision += precision
                num_cycles += 1

    final_precision = total_precision / num_cycles
    execution_time = time.time() - start_time

    print('Final Top N Precision: %f' % final_precision)
    print("--- %s seconds ---" % execution_time)

    result = {
        'Top N': final_precision,
        'Execution time': execution_time
    }

    return result


def calculate_recommender_precision(
        test_data, user_id, recommender, n, min_score):

    known_ratings = extractor.get_user_item_ratings(test_data, user_id, True)
    items = known_ratings.keys()

    predicted_ratings = {}

    for item in items:
        predicted_ratings[item] = recommender.predict_rating(user_id, item)

    return calculate_precision(known_ratings, predicted_ratings, n, min_score)


def calculate_precision(known_ratings, predicted_ratings, n, min_score):

    if not known_ratings:
        return None

    sorted_predicted_ratings = sorted(
        predicted_ratings.iteritems(),
        key=operator.itemgetter(1), reverse=True)[:n]

    num_hits = 0.
    for item, predicted_rating in sorted_predicted_ratings:
        known_rating = known_ratings[item]

        if known_rating >= min_score:
            num_hits += 1

    return num_hits / min(n, len(known_ratings))


def calculate_recall_in_top_n(
        records, recommender, n, num_folds, split=None, min_score=5.0,
        cache_reviews=None, reviews_type=None):

    start_time = time.time()
    if split is None:
        split = 1 - (1/float(num_folds))
    # split = 0.984
    total_recall = 0.
    total_coverage = 0.
    num_cycles = 0.0
    num_folds = 1

    for i in xrange(0, num_folds):
        print('Fold', i)
        print('started training', time.strftime("%Y/%d/%m-%H:%M:%S"))
        start = float(i) / num_folds
        cluster_labels = None
        train_records, test_records = ETLUtils.split_train_test(
            records, split=split, shuffle_data=False, start=start)
        if cache_reviews:
            train_reviews, test_reviews = ETLUtils.split_train_test(
                cache_reviews, split=split, shuffle_data=False, start=start)
            if reviews_type is not None:
                cluster_labels = reviews_clusterer.cluster_reviews(test_reviews)
            recommender.reviews = train_reviews
        recommender.load(train_records)

        print('finished training', time.strftime("%Y/%d/%m-%H:%M:%S"))

        if cluster_labels is not None:
            separated_records = reviews_clusterer.split_list_by_labels(
                test_records, cluster_labels)
            if reviews_type == 'specific':
                test_records = separated_records[0]
            if reviews_type == 'generic':
                test_records = separated_records[1]

        positive_reviews = \
            [review for review in test_records if review['overall_rating'] >= min_score]

        if len(positive_reviews) == 0:
            continue

        num_hits = 0.0
        num_predictions = 0.0
        for review in positive_reviews:
            user_id = review['user_id']
            item_id = review['offering_id']
            if not recommender.has_context:
                hit = calculate_is_a_hit(
                    test_records, recommender, user_id, item_id, n)
            else:
                text_review = review['text']
                hit = calculate_is_a_hit(
                    test_records, recommender, user_id, item_id, n, text_review)
            if hit is None:
                continue
            if hit:
                num_hits += 1
            num_predictions += 1
            # print('num predictions: %d/%d %s' % (num_predictions, len(positive_reviews), time.strftime("%Y/%d/%m-%H:%M:%S")))

        if num_predictions == 0:
            continue

        recommender.clear()
        recall = num_hits / num_predictions
        coverage = num_predictions / len(positive_reviews)
        print('recall', recall, time.strftime("%Y/%d/%m-%H:%M:%S"))
        print('coverage', coverage, time.strftime("%Y/%d/%m-%H:%M:%S"))
        total_recall += recall
        total_coverage += coverage
        num_cycles += 1

    final_recall = total_recall / num_cycles
    final_coverage = total_coverage / num_cycles
    execution_time = time.time() - start_time

    print('Final Top N Precision: %f' % final_recall)
    print('Final Coverage: %f' % final_coverage)
    print("--- %s seconds ---" % execution_time)

    result = {
        'Top N': final_recall,
        'Coverage': final_coverage,
        'Execution time': execution_time
    }

    return result


def calculate_is_a_hit(reviews, recommender, user_id, liked_item, n, text_review=None):
    unknown_items = get_unknown_items(reviews, user_id)
    # unknown_items.append(liked_item)
    # all_items = unknown_items[:]

    if not recommender.has_context:
        desired_prediction =\
            recommender.predict_rating(user_id, liked_item)
    else:
        desired_prediction =\
            recommender.predict_rating(user_id, liked_item, text_review)

    if desired_prediction is None:
        return None

    predicted_ratings = {liked_item: desired_prediction}

    for item in unknown_items:
        if not recommender.has_context:
            predicted_ratings[item] = recommender.predict_rating(user_id, item)
        else:
            predicted_ratings[item] =\
                recommender.predict_rating(user_id, item, text_review)

    return is_a_hit(liked_item, predicted_ratings, n)


def get_unknown_items(reviews, user_id, num_unknown=1000):
    item_ids = extractor.get_groupby_list(reviews, 'offering_id')
    user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user_id])
    user_items = extractor.get_groupby_list(user_reviews, 'offering_id')

    # We calculate which are the items that the user hasn't rated, which is the
    # items that are in the list item_ids but not in the list user_items
    s = set(user_items)
    unknown_items = [x for x in item_ids if x not in s]
    # TODO: Uncomment this line, the items have to be shuffled
    # shuffle(unknown_items)

    return unknown_items[:num_unknown]


def is_a_hit(preferred_item, predicted_ratings, n):

    sorted_predicted_ratings = sorted(
        predicted_ratings.iteritems(),
        key=operator.itemgetter(1),
        reverse=True)[:n]

    for predicted_rating in sorted_predicted_ratings:
        if preferred_item == predicted_rating[0]:
            return True
    return False
