from collections import Counter
import json
import math
# from sets import Set
import random
import string
import nltk
from nltk.corpus import wordnet
# from nltk.corpora import wordnet as wn
from nltk.corpus.reader import Synset
import numpy as np
import re
from sklearn.cluster import KMeans
import time
from topicmodeling.context.senses_group import SenseGroup
from topicmodeling.context.review import Review

__author__ = 'fpena'

from nltk import tokenize


def count_sentences(text):
    """
    Returns the number of sentences there are in the given text

    :type text: str
    :param text: just a text
    :rtype: int
    :return: the number of sentences there are in the given text
    """
    return len(tokenize.sent_tokenize(text))


def count_words(text):
    """
    Returns the number of words there are in the given text

    :type text: str
    :param text: just a text. It must be in english.
    :rtype: int
    :return: the number of words there are in the given text
    """
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)

    words = []

    for sentence in sentences:
        words.extend([word.strip(string.punctuation) for word in sentence.split()])
    return math.log(len(words) + 1)


def vbd_sum(tags_count):

    return math.log(tags_count['VBD'] + 1)


def verb_sum(tags_count):

    total_verbs =\
        tags_count['VB'] + tags_count['VBD'] + tags_count['VBG'] +\
        tags_count['VBN'] + tags_count['VBP'] + tags_count['VBZ']

    return math.log(total_verbs + 1)


def process_review(review):
    """

    :type review: Review
    :param review:
    :return:
    """
    log_sentence = math.log(count_sentences(review.text) + 1)
    log_word = math.log(count_words(review.text) + 1)
    tagged_words = review.tagged_words
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


def cluster_reviews(reviews):
    """

    :param reviews:
    :type reviews: list[Review]
    """

    records = np.zeros((len(reviews), 5))

    for index in range(len(reviews)):
        records[index] = process_review(reviews[index])

    # print('processed records', time.strftime("%H:%M:%S"))

    k_means = KMeans(n_clusters=2)
    k_means.fit(records)
    labels = k_means.labels_

    record_clusters = split_list_by_labels(records, labels)
    cluster0_sum = reduce(lambda x, y: x + sum(y), record_clusters[0], 0)
    cluster1_sum = reduce(lambda x, y: x + sum(y), record_clusters[1], 0)

    if cluster0_sum < cluster1_sum:
        # If the cluster 0 contains the generic review we invert the tags
        labels = [1 if element == 0 else 0 for element in labels]

    print('clustered reviews', time.strftime("%H:%M:%S"))

    return labels

    # review_clusters = split_list_by_labels(text_reviews, labels)
    #
    # if cluster0_sum > cluster1_sum:
    #     specific_reviews = review_clusters[0]
    #     generic_reviews = review_clusters[1]
    # else:
    #     specific_reviews = review_clusters[1]
    #     generic_reviews = review_clusters[0]
    #
    # return specific_reviews, generic_reviews


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

    print('building groups', time.strftime("%H:%M:%S"))
    all_senses = set()

    sense_word_map = {}
    for noun in nouns:
        senses = wordnet.synsets(noun, pos='n')
        all_senses.update(senses)
        for sense in senses:
            if sense not in sense_word_map:
                sense_word_map[sense] = []
            sense_word_map[sense].append(noun)

    all_senses = list(all_senses)


    print('number of senses:', len(all_senses))
    senses_similarity_matrix = build_sense_similarity_matrix(all_senses)

    groups = []
    bronk2_synset([], all_senses[:], [], groups, senses_similarity_matrix)
    # bronk2_synset([], all_senses[:], [], groups, all_senses[:])

    sense_groups = []
    for group in groups:
        sense_group = SenseGroup(group)
        for sense in sense_group.senses:
            sense_group.nouns |= set(sense_word_map[sense])
        sense_groups.append(sense_group)

    print('number of sense groups:', len(sense_groups))

    print('finished groups', time.strftime("%H:%M:%S"))

    return sense_groups


def build_sense_similarity_matrix(senses):
    """

    :type senses: list[Synset]
    :param senses:
    """
    print('building senses similarity matrix', time.strftime("%H:%M:%S"))
    similarity_matrix = {}

    for sense in senses:
        similarity_matrix[sense] = {}

    index = 1
    for sense1 in senses:
        for sense2 in senses[index:]:
            similarity = sense1.wup_similarity(sense2)
            similarity_matrix[sense1][sense2] = similarity
            similarity_matrix[sense2][sense1] = similarity
        index += 1

    print('finished senses similarity matrix', time.strftime("%H:%M:%S"))

    return similarity_matrix


def get_synset_neighbours(synset, similarity_matrix):
    neighbours = []

    # for element in potential_neighbours:
    #     if synset == element:
    #         continue
    #
    #     # print('synset', synset, 'element', element, 'similarity', synset.wup_similarity(element))
    #     if synset.wup_similarity(element) >= 0.9:
    #         neighbours.append(element)

    # for element in similarity_matrix:
    for element in similarity_matrix.keys():
        if synset == element:
            continue

        if similarity_matrix[synset][element] >= 0.7:
            neighbours.append(element)
        # if synset.wup_similarity(element) >= 0.9:
        #     neighbours.append(element)

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
        if not frozenset(review.senses).isdisjoint(frozenset(group.senses)):
            num_reviews += 1

    return num_reviews / len(reviews)


def get_context_similarity(context1, context2, topic_indices):

    # We filter the topic model, selecting only the topics that contain context
    filtered_context1 = np.array([context1[i[0]] for i in topic_indices])
    filtered_context2 = np.array([context2[i[0]] for i in topic_indices])

    return 1 / (1 + np.linalg.norm(filtered_context1-filtered_context2))


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


def bronk2_synset(clique, candidates, excluded, clique_list, similarity_matrix):
    if len(candidates) == 0 and len(excluded) == 0:
        # print clique
        clique_list.append(clique)
        return
    pivot = choose_pivot(candidates, excluded)
    neighbours = get_synset_neighbours(pivot, similarity_matrix)
    p_minus_neighbours = list_difference(candidates, neighbours)[:]
    for vertex in p_minus_neighbours:
        vertex_neighbours = get_synset_neighbours(vertex, similarity_matrix)
        new_candidates = [val for val in candidates if val in vertex_neighbours]  # p intersects N(vertex)
        new_excluded = [val for val in excluded if val in vertex_neighbours]  # x intersects N(vertex)
        bronk2_synset(clique + [vertex], new_candidates, new_excluded, clique_list, similarity_matrix)
        candidates.remove(vertex)
        excluded.append(vertex)

def generate_stats(specific_reviews, generic_reviews):

    num_specific = float(len(specific_reviews))
    num_generic = float(len(generic_reviews))
    num_total_reviews = num_specific + num_generic

    print('Specific reviews: %d (%f %%)' % (num_specific, (num_specific / num_total_reviews * 100)))
    stat_reviews(specific_reviews)

    print('Generic reviews %d (%f %%)' % (num_generic, (num_generic / num_total_reviews * 100)))
    stat_reviews(generic_reviews)

def stat_reviews(reviews):
    """

    :type reviews: list[Review]
    :param reviews:
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    stats = np.zeros(5)
    num_reviews = len(reviews)
    for review in reviews:
        text = review.text
        num_sentences = len(tokenize.sent_tokenize(text))
        num_words = len(tokenizer.tokenize(text.lower()))
        tagged_words = review.tagged_words
        tags_count = Counter(tag for word, tag in tagged_words)
        num_past_verbs = float(tags_count['VBD'])
        num_verbs = tags_count['VB'] + tags_count['VBD'] + tags_count['VBG'] +\
            tags_count['VBN'] + tags_count['VBP'] + tags_count['VBZ']
        ratio = (num_past_verbs + 1) / (num_verbs + 1)

        stats[0] += num_sentences
        stats[1] += num_words
        stats[2] += num_past_verbs
        stats[3] += num_verbs
        stats[4] += ratio

    for index in range(len(stats)):
        stats[index] /= num_reviews

    print('Average sentences:', stats[0])
    print('Average words:', stats[1])
    print('Average past verbs:', stats[2])
    print('Average verbs:', stats[3])
    print('Average past verbs ratio:', stats[4])




# my_tokens = nltk.word_tokenize(review_text7)
# my_tags = nltk.pos_tag(my_tokens)
#
# for word, tag in my_tags:
#     print("%s\t%s" % (word, tag))
# print("\n************\n")
# for word, tag in tag_words(review_text7):
#     print("%s\t%s" % (word, tag))
#
# my_review = re.sub("\s\s+", " ", review_text7)
# my_result = [word.strip(string.punctuation) for word in my_review.split(" ")]
#
# print(my_result)
# print(nltk.pos_tag(my_result))
#
# my_sentence_tokenizer = nltk.data.load(
#             'tokenizers/punkt/english.pickle')
# for sentence in my_sentence_tokenizer.tokenize(my_review):
#     print(sentence)

# Split the words in sentences
# Lower case the first word of the sentence
# split the words by whitespace and then remove leading and trailing punctuation
#
#

# print(len(list(wordnet.all_synsets('n'))))

# my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]


my_text1 = "BUT if you do stay here, it's awesome."
my_text2 = "great hotel in Central Phoenix for a staycation, but not " \
           "necessarily a place to stay out of town and without a car. " \
           "Not much around the area, and unless you're familiar with " \
           "downtown, I would rather have a guest stay in " \
           "Old Town Scottsdale, etc. BUT if you do stay here, it's awesome." \
           " Great boutique rooms. Awesome pool that's happening in the " \
           "summer. A GREAT rooftop patio bar, and a very very busy lobby " \
           "with Gallo Blanco attached. A great place to stay, but have a car!"
my_tokens1 = nltk.word_tokenize(my_text1)
my_tokens2 = nltk.word_tokenize(my_text2)
my_tags1 = nltk.pos_tag(my_tokens1)
my_tags2 = nltk.pos_tag(my_tokens2)
# print(my_tags1)
# print(my_tags2)
#
# from nltk.tag.stanford import POSTagger
# st = POSTagger("/Users/fpena/tmp/stanford-postagger-full-2015-04-20/models/english-left3words-distsim.tagger",
#                "/Users/fpena/tmp/stanford-postagger-full-2015-04-20/stanford-postagger.jar")
# print(st.tag(my_tokens2))
