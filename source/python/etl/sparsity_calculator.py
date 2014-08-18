from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'


def get_sparsity(reviews):
    """
    Returns the percentage of missing ratings in the given list of reviews

    :param reviews: a list of dictionaries, in which each dictionary represents
    a review and must contain the fields 'user_id' for the user and
    'offering_id' for the item
    :return: the rate of missing ratings
    (i.e. number of missing ratings / (number of items * number of users))
    :raise ValueError: in case an empty list is given
    """
    if not reviews:
        raise ValueError('Can not determine the sparsity for an empty list')

    user_ids = extractor.get_groupby_list(reviews, 'user_id')
    item_ids = extractor.get_groupby_list(reviews, 'offering_id')

    non_missing_reviews = 0.
    total_expected_reviews = len(user_ids) * len(item_ids)

    for user in user_ids:
        user_reviews = ETLUtils.filter_records(reviews, 'user_id', [user])
        user_items = extractor.get_groupby_list(user_reviews, 'offering_id')

        non_missing_reviews += len(set(item_ids).intersection(set(user_items)))

    return 1 - non_missing_reviews / total_expected_reviews