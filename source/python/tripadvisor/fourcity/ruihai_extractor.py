from etl import ETLUtils
from tripadvisor.fourcity import extractor

__author__ = 'fpena'

# First thing to do is convert the XML into JSON

import xmltodict


def load_reviews(file_name):

    document_file = open(file_name, "r")  # Open a file in read-only mode
    original_doc = document_file.read()  # read the file object
    document = xmltodict.parse(original_doc)  # Parse the read document string

    raw_reviews = document['REVIEWS']['REVIEW_LIST']['REVIEW']

    processed_reviews = []

    for raw_review in raw_reviews:
        processed_review = {}
        processed_review['user_id'] = raw_review['MBR_NAME']
        processed_review['offering_id'] = raw_review['HOTEL_ID']
        processed_review['overall_rating'] = float(raw_review['RATING'])

        processed_reviews.append(processed_review)

    return processed_reviews


def load_all_reviews():
    city_files = [
        '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/Chicago_review.xml',
        '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/Dublin_review.xml',
        '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/Hong kong_review.xml',
        '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/London_review.xml',
        '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/New York_review.xml',
        '/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/Singapore_review.xml'
    ]

    all_reviews = []

    for city_file in city_files:
        city_reviews = load_reviews(city_file)
        all_reviews.extend(city_reviews)

    ETLUtils.save_json_file('/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/all_reviews.json', all_reviews)

    cleaned_reviews = clean_reviews(all_reviews)
    ETLUtils.save_json_file('/Users/fpena/UCC/Thesis/datasets/TripHotelReviewXml/cleaned_reviews.json', cleaned_reviews)

    return all_reviews

def clean_reviews(reviews):
    """
    Returns a copy of the original reviews list with only that are useful for
    recommendation purposes

    :param reviews: a list of reviews
    :return: a copy of the original reviews list with only that are useful for
    recommendation purposes
    """
    filtered_reviews = extractor.remove_empty_user_reviews(reviews)
    print('Finished remove_missing_ratings_reviews')
    filtered_reviews = extractor.remove_users_with_low_reviews(filtered_reviews, 5)
    print('Finished remove_users_with_low_reviews')
    # filtered_reviews = extractor.remove_items_with_low_reviews(filtered_reviews, 5)
    # print('Finished remove_single_review_hotels')
    # filtered_reviews = remove_users_with_low_reviews(filtered_reviews, 10)
    # print('Finished remove_users_with_low_reviews')
    print('Number of reviews', len(filtered_reviews))
    return filtered_reviews

reviews = load_all_reviews()
# print('Total reviews', len(reviews))
# my_reviews = clean_reviews(reviews)
# print('Total reviews', len(my_reviews))
#
# reviewsDatasetAnalyzer = ReviewsDatasetAnalyzer(reviews)
# print(reviewsDatasetAnalyzer.calculate_sparsity())
# common_item_counts = reviewsDatasetAnalyzer.count_items_in_common()
# print(common_item_counts)
# print(reviewsDatasetAnalyzer.analyze_common_items_count(common_item_counts))
# print(reviewsDatasetAnalyzer.analyze_common_items_count(common_item_counts, True))

