from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'

def clean_reviews(reviews):
    """
    Returns a copy of the original reviews list with only that are useful for
    recommendation purposes

    :param reviews: a list of reviews
    :return: a copy of the original reviews list with only that are useful for
    recommendation purposes
    """
    # filtered_reviews = remove_empty_user_reviews(reviews)
    # filtered_reviews = remove_missing_ratings_reviews(filtered_reviews)
    # print('Finished remove_missing_ratings_reviews')
    filtered_reviews = extractor.remove_users_with_low_reviews(reviews, 10)
    print('Finished remove_users_with_low_reviews')
    filtered_reviews = extractor.remove_items_with_low_reviews(filtered_reviews, 20)
    print('Finished remove_single_review_hotels')
    filtered_reviews = extractor.remove_users_with_low_reviews(filtered_reviews, 10)
    filtered_reviews = extractor.remove_items_with_low_reviews(filtered_reviews, 20)
    filtered_reviews = extractor.remove_users_with_low_reviews(filtered_reviews, 10)
    filtered_reviews = extractor.remove_items_with_low_reviews(filtered_reviews, 20)
    filtered_reviews = extractor.remove_users_with_low_reviews(filtered_reviews, 10)
    filtered_reviews = extractor.remove_items_with_low_reviews(filtered_reviews, 20)
    filtered_reviews = extractor.remove_users_with_low_reviews(filtered_reviews, 10)
    filtered_reviews = extractor.remove_items_with_low_reviews(filtered_reviews, 20)
    print('Finished remove_users_with_low_reviews')
    print('Number of reviews', len(filtered_reviews))
    return filtered_reviews
    # pass


def pre_process_reviews():
    """
    Returns a list of preprocessed reviews, where the reviews have been filtered
    to obtain only relevant data, have dropped any fields that are not useful,
    and also have additional fields that are handy to make calculations

    :return: a list of preprocessed reviews
    """
    reviews_file = '/Users/fpena/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json'
    reviews = ETLUtils.load_json_file(reviews_file)

    select_fields = ['user_id', 'business_id', 'stars']
    reviews = ETLUtils.select_fields(select_fields, reviews)
    extract_fields(reviews)
    ETLUtils.drop_fields(['business_id', 'stars'], reviews)
    # reviews = load_json_file('/Users/fpena/tmp/filtered_reviews.json')
    reviews = clean_reviews(reviews)

    return reviews

def extract_fields(reviews):
    """
    Modifies the given list of reviews in order to extract the values contained
    in the ratings field to top level fields. For instance, a review which is
    in the form
    {'user_id': 'U1', 'offering_id': :'I1',
    'ratings': {'cleanliness': 4.0, 'location': 5.0}}
    would become:

    {'user_id': 'U1', 'offering_id': :'I1',
    'ratings': {'cleanliness': 4.0, 'location': 5.0},
    'cleanliness_rating': 4.0, 'location_rating': 5.0}

    :param reviews: a list of reviews.
    """

    for review in reviews:
        review['offering_id'] = review['business_id']
        review['overall_rating'] = review['stars']


my_reviews = pre_process_reviews()
filtered_reviews_file = '/Users/fpena/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/filtered_reviews.json'
ETLUtils.save_json_file(filtered_reviews_file, my_reviews)
# print('Num reviews', len(my_reviews))
print(my_reviews[0])
print(my_reviews[1])
# print(my_reviews[2])
# print(my_reviews[3])