import json

from topicmodeling.context.review import Review

__author__ = 'fpena'


USER_ID_FIELD = 'user_id'
ITEM_ID_FIELD = 'item_id'
RATING_FIELD = 'rating'

# Optional fields
TEXT_FIELD = 'text'
REVIEW_ID = 'review_id'
DATE = 'date'

# Generated fields
REVIEW_TYPE = 'type'
TAGGED_WORDS = 'tagged_words'
NOUNS = 'nouns'
TOPICS = 'topics'


def load_review(record):
    keys = record.keys()

    review = Review()
    review.user_id = record[USER_ID_FIELD]
    review.item_id = record[ITEM_ID_FIELD]
    review.rating = record[RATING_FIELD]

    # Optional fields
    if TEXT_FIELD in keys:
        review.text = record[TEXT_FIELD]
    if REVIEW_ID in keys:
        review.id = record[REVIEW_ID]
    if DATE in keys:
        review.date = record[DATE]

    # Generated fields
    if REVIEW_TYPE in keys:
        review.type = record[REVIEW_TYPE]
    if TAGGED_WORDS in keys:
        review.tagged_words = record[TAGGED_WORDS]
    if NOUNS in keys:
        review.nouns = record[NOUNS]
    if TOPICS in keys:
        review.topics = record[TOPICS]

    return review


def load_review_yelp_recsys_2013(record):

    user_id_field = 'user_id'
    item_id_field = 'business_id'
    rating_field = 'stars'
    text_field = 'text'
    review_id = 'review_id'
    date = 'date'

    # Generated fields
    review_type = 'predicted_class'
    tagged_words = 'tagged_words'
    nouns = 'nouns'
    topics = 'topics'

    keys = record.keys()

    review = Review()
    review.user_id = record[user_id_field]
    review.item_id = record[item_id_field]
    review.rating = record[rating_field]

    # Optional fields
    if TEXT_FIELD in keys:
        review.text = record[text_field]
    if REVIEW_ID in keys:
        review.id = record[review_id]
    if DATE in keys:
        review.date = record[date]

    # Generated fields
    if REVIEW_TYPE in keys:
        review.type = record[review_type]
    if TAGGED_WORDS in keys:
        review.tagged_words = record[tagged_words]
    if NOUNS in keys:
        review.nouns = record[nouns]
    if TOPICS in keys:
        review.topics = record[topics]

    return review


def load_reviews_yelp_recsys_2013(file_path):
    reviews = []
    for line in open(file_path):
        reviews.append(load_review_yelp_recsys_2013(json.loads(line)))
    return reviews


def load_reviews(file_path):
    reviews = []
    for line in open(file_path):
        reviews.append(load_review(json.loads(line)))
    return reviews


def save_reviews(file_path, reviews):
    with open(file_path, 'w') as outfile:
        for review in reviews:
            json.dump(review, outfile, default=lambda o: o.__dict__)
            outfile.write('\n')


def get_user_ids(reviews):
    user_ids = set()
    for review in reviews:
        user_ids.add(review.user_id)
    return user_ids


def get_item_ids(reviews):
    item_ids = set()
    for review in reviews:
        item_ids.add(review.item_id)
    return item_ids



# DATASET = 'hotel'
DATASET = 'restaurant'
DATASET_FOLDER = '/Users/fpena/UCC/Thesis/datasets/context/'
OUTPUT_FOLDER = DATASET_FOLDER + '2nd_generation/'
REVIEWS_FILE = OUTPUT_FOLDER + 'reviews_' + DATASET + '_shuffled.json'
RECORDS_FILE = DATASET_FOLDER + 'yelp_training_set_review_' +\
               DATASET + 's_shuffled_tagged.json'
# REVIEWS_FILE = DATASET_FOLDER + 'reviews_' + DATASET + '_shuffled.pkl'

# start = time.time()
# my_reviews = load_reviews(REVIEWS_FILE)
# print(my_reviews[0].tagged_words[0])
# print(my_reviews[0].tagged_words[0][0])
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

# start = time.time()
# my_records = ETLUtils.load_json_file(RECORDS_FILE)
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
#
# start = time.time()
# my_reviews = load_reviews(RECORDS_FILE)
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
#
#
#
# my_review = load_review(my_records[0])
# print(my_records[0])
# print(my_review.__dict__)
# # my_review2 = json.loads(my_records[0], object_pairs_hook=load_review)
#
# print(json.dumps(my_review.__dict__))
# print(json.dumps(my_review, default=lambda o: o.__dict__))
#
# tmp_file = '/Users/fpena/tmp/tmp_file.json'
#
# my_string = json.dumps(my_review, default=lambda o: o.__dict__)
# my_review2 = json.loads(my_string)
# my_string2 = json.dumps(my_review2, default=lambda o: o.__dict__)
# my_review3 = json.loads(my_string2)
#
#
# print(my_review2)
# print(my_review3)
# print(my_review2 == my_review3)
#
#
# start = time.time()
# user_ids1 = extractor.get_groupby_list(my_records, 'user_id')
# print(len(user_ids1))
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
#
# start = time.time()
# user_ids2 = get_user_ids(my_reviews)
# print(len(user_ids2))
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

# new_reviews = load_reviews_yelp_recsys_2013(RECORDS_FILE)
# with open(REVIEWS_FILE, 'rb') as read_file:
#     old_reviews = pickle.load(read_file)

# for new_review, old_review in zip(new_reviews, old_reviews):
#     if new_review.text != old_review.text:
#         print('Something is wrong...')
#     new_review.tagged_words = old_review.tagged_words
# save_reviews(OUTPUT_FILE, new_reviews)
