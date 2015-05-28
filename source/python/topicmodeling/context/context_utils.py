from collections import Counter
import json
import math
# from sets import Set
import random
import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset
import numpy as np
from sklearn.cluster import KMeans
import time
from topicmodeling.context.senses_group import SenseGroup

__author__ = 'fpena'

from nltk import tokenize


def log_sentences(text):
    return math.log(len(tokenize.sent_tokenize(text)) + 1)


def log_words(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    return math.log(len(tokens) + 1)


def tag_words(text):
    # tokens = nltk.word_tokenize(text.lower())
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text = text.encode('utf-8')
    tokens = tokenizer.tokenize(text.lower())
    nltk_text = nltk.Text(tokens)
    tagged_words = nltk.pos_tag(nltk_text)
    # sentences = nltk.sent_tokenize(text)
    # sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # tagged_words = []
    # for sent in sentences:
    #     tagged_words.extend(nltk.pos_tag(sent))
    # sentences = [nltk.pos_tag(sent) for sent in sentences]
    # print(tagged_words)
    return tagged_words


def vbd_sum(tags_count):

    return math.log(tags_count['VBD'] + 1)


def verb_sum(tags_count):

    total_verbs =\
        tags_count['VB'] + tags_count['VBD'] + tags_count['VBG'] +\
        tags_count['VBN'] + tags_count['VBP'] + tags_count['VBZ']

    return math.log(total_verbs + 1)


def process_review(review):
    log_sentence = log_sentences(review)
    log_word = log_words(review)
    tagged_words = tag_words(review)
    # print(tagged_words)
    counts = Counter(tag for word, tag in tagged_words)
    log_past_verbs = vbd_sum(counts)
    log_verbs = verb_sum(counts)
    # log_past_verbs = 1
    # log_verbs = 1

    # This ensures that when log_verbs = 0 the program won't crash
    if log_past_verbs == log_verbs:
        verbs_ratio = 1
    else:
        verbs_ratio = log_past_verbs / log_verbs

    result = [log_sentence, log_word, log_past_verbs, log_verbs, verbs_ratio]

    return np.array(result)


def cluster_reviews(text_reviews):
    """

    :type text_reviews: list[str]
    """

    records = np.zeros((len(text_reviews), 5))

    for index in range(len(text_reviews)):
        records[index] = process_review(text_reviews[index])

    print('processed records', time.strftime("%H:%M:%S"))

    k_means = KMeans(n_clusters=2)
    k_means.fit(records)
    labels = k_means.labels_
    print('clustered reviews', time.strftime("%H:%M:%S"))

    record_clusters = split_list_by_labels(records, labels)
    cluster0_sum = reduce(lambda x, y: x + sum(y), record_clusters[0], 0)
    cluster1_sum = reduce(lambda x, y: x + sum(y), record_clusters[1], 0)

    review_clusters = split_list_by_labels(text_reviews, labels)

    if cluster0_sum > cluster1_sum:
        specific_reviews = review_clusters[0]
        generic_reviews = review_clusters[1]
    else:
        specific_reviews = review_clusters[1]
        generic_reviews = review_clusters[0]

    return specific_reviews, generic_reviews


def split_list_by_labels(lst, labels):

    matrix = []

    for index in range(max(labels) + 1):
        matrix.append([])

    for index in range(len(labels)):
        element = lst[index]
        matrix[labels[index]].append(element)

    return matrix


def load_reviews(reviews_file):

    records = [json.loads(line) for line in open(reviews_file)]
    reviews = []

    for record in records:
        reviews.append(record['text'])

    return reviews


def get_nouns(word_tags):
    return [word for (word, tag) in word_tags if tag.startswith('N')]


def get_all_nouns(reviews):

    nouns = set()

    for review in reviews:
        nouns |= set(review.nouns)

    return nouns


def remove_nouns_from_reviews(reviews, nouns):

    for noun in nouns:
        for review in reviews:
            if noun in review.nouns:
                review.nouns.remove(noun)


def generate_senses(review):

    review.senses = set()
    for noun in review.nouns:
        review.senses |= set(wordnet.synsets(noun, pos='n'))


def generate_all_senses(reviews):

    all_senses = set()

    for review in reviews:
        generate_senses(review)
        all_senses |= set(review.senses)

    return all_senses


def calculate_word_weighted_frequency(word, reviews):
    """

    :param self:
    :type word: str
    :param word:
    :type reviews: list[Review]
    :param reviews:
    :rtype: float
    :return:
    """
    num_reviews = 0.0

    for review in reviews:
        if word in review.nouns:
            num_reviews += 1

    return num_reviews / len(reviews)


def build_groups(nouns):

    all_senses = set()

    for noun in nouns:
        all_senses.update(wordnet.synsets(noun, pos='n'))

    # print(all_senses)
    all_senses = list(all_senses)

    groups = []
    bronk2_synset([], all_senses[:], [], groups, all_senses[:])

    sense_groups = []
    for group in groups:
        sense_groups.append(SenseGroup(group))

    return sense_groups


def get_synset_neighbours(synset, potential_neighbours):
    """

    :rtype : object
    :type synset: Synset
    :param synset:
    :param potential_neighbours:
    :return:
    """
    neighbours = []

    for element in potential_neighbours:
        if synset == element:
            continue

        # print('synset', synset, 'element', element, 'similarity', synset.wup_similarity(element))
        if synset.wup_similarity(element) >= 0.9:
            neighbours.append(element)

    return neighbours


def is_similar(number1, number2):
    if math.fabs(number1 - number2) < 3:
        return True
    return False


def get_neighbours(number, potential_neighbours):

    neighbours = []

    for element in potential_neighbours:
        if number == element:
            continue
        if is_similar(number, element):
            neighbours.append(element)

    return neighbours


def is_noun_in_group(noun, group):
    senses = wordnet.synsets(noun, pos='n')
    return any(i in senses for i in group)


def is_group_in_review(group, review):
    """

    :type group list[Synset]
    :param group:
    :type review: Review
    :param review:
    """

    for noun in review.nouns:
        if is_noun_in_group(noun, group):
            return True

    return False


def get_text_from_reviews(reviews):
    """
    Receives a list[Review] and extracts the text contained in each review.
    Returns a list[str].

    :type reviews: list[Review]
    :param reviews:
    """
    text_reviews = []
    for review in reviews:
        text_reviews.append(review.text)

    return text_reviews


def calculate_group_weighted_frequency(group, reviews):
    num_reviews = 0.0

    for review in reviews:
        if frozenset(review.senses).isdisjoint(frozenset(group.senses)):
            num_reviews += 1

    return num_reviews / len(reviews)


def main():
    # reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review-short.json"
    reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review-mid.json"
    reviews = load_reviews(reviews_file)
    print("reviews:", len(reviews))
    specific, generic = cluster_reviews(reviews)

    print(specific)
    print(generic)

# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)



def choose_pivot(list1, list2):

    index = random.randint(0, len(list1) + len(list2) - 1)
    if index < len(list1):
        return list1[index]
    else:
        return list2[index-len(list1)]


def list_difference(list1, list2):
    return [item for item in list1 if item not in list2]


def bronk2_synset(clique, candidates, excluded, clique_list, synsets):
    if len(candidates) == 0 and len(excluded) == 0:
        # print clique
        clique_list.append(clique)
        return
    pivot = choose_pivot(candidates, excluded)
    neighbours = get_synset_neighbours(pivot, synsets)
    p_minus_neighbours = list_difference(candidates, neighbours)[:]
    for vertex in p_minus_neighbours:
        vertex_neighbours = get_synset_neighbours(vertex, synsets)
        new_candidates = [val for val in candidates if val in vertex_neighbours]  # p intersects N(vertex)
        new_excluded = [val for val in excluded if val in vertex_neighbours]  # x intersects N(vertex)
        bronk2_synset(clique + [vertex], new_candidates, new_excluded, clique_list, synsets)
        candidates.remove(vertex)
        excluded.append(vertex)
