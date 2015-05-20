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
    print(tagged_words)
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

    :type reviews: list[str]
    """

    records = np.zeros((len(reviews), 5))

    for index in range(len(reviews)):
        records[index] = process_review(reviews[index])

    print('processed records')

    k_means = KMeans(n_clusters=2)
    k_means.fit(records)
    labels = k_means.labels_
    print('clustered reviews')

    return labels


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


def calculate_weighted_frequency(word, reviews):
        """

        :param self:
        :type word: str
        :param word:
        :type reviews: list[Review]
        :param reviews:
        :rtype: float
        :return:
        """
        num_reviews = 0

        for review in reviews:
            if word in review.nouns:
                num_reviews += 1

        return num_reviews / len(reviews)


def build_groups(nouns):

    all_senses = set()

    for noun in nouns:
        all_senses.update(wordnet.synsets(noun, pos='n'))

    groups = []
    bronk2_synset([], all_senses[:], [], groups, all_senses[:])

    return groups

    return groups


def get_synset_neigbours(synset, potential_neighbours):
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
        if synset.wup_similarity(element) >= 0.9:
            neighbours.append(element)

    return neighbours


def build_groups_n(list, similarity_set):

    if not list:
        return [similarity_set]

    # first_element = list[0]
    # add_group(first_element, similarity_set)
    for element in list:
        add_group(element, similarity_set)

    if len(list) == 1:
        return [similarity_set]

    list.pop(0)
    copy_set = similarity_set.copy()
    copy_list = list[:]

    return [] + build_groups_n(copy_list, copy_set) + build_groups_n(copy_list, set())

def build_groups_n2(list, similarity_set):

    if not list:
        return [similarity_set]

    all_groups = set()

    for element_i in list:
        group = set([element_i])
        for element_j in list:
            add_group(element_j, group)
        all_groups.add(frozenset(group))

    return all_groups

def build_groups_n3(elements):

    for element in elements:
        neighbours = get_neighbours(element, elements)


def add_group(number, group):
    for element in group:
        if not is_similar(element, number):
            return
    group.add(number)


def is_similar(number1, number2):
    if math.fabs(number1 - number2) < 3:
        return True
    return False


def get_neighbours(number, potential_neigbours):

    neighbours = []

    for element in potential_neigbours:
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


def main():
    reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review-short.json"
    # reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review-mid.json"
    reviews = load_reviews(reviews_file)
    print("reviews:", len(reviews))
    labels = cluster_reviews(reviews)

    print(labels)



from sklearn import datasets

iris = datasets.load_iris()
# X = iris.data
# X = [[0, 1], [0, 2], [3, 3], [0, 1]]
X = np.zeros((4, 2))
X[0] = [0, 1]
X[1] = [0, 2]
X[2] = [3, 3]
X[3] = [0, 1]
my_k_means = KMeans(n_clusters=2)
my_k_means.fit(X)
labels = my_k_means.labels_

# print(labels)

# print(X)
# print(X[1])
# print(X[2])







# p = "Good morning Dr. Adams. The patient is waiting for you in room number 3."
# #
# my_text = "We had dinner there last night. The food was delicious. Definitely, is the best restaurant in town."
# my_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
# my_tokens = my_tokenizer.tokenize(my_text.lower())
# my_tokens = nltk.word_tokenize(my_text.lower())
# mtext = nltk.Text(my_tokens)
# my_tags = nltk.pos_tag(mtext)
#
# print(my_tags)
# my_counts = Counter(tag for word, tag in my_tags)
# print(my_counts)
# print(my_counts['VBNZ'])

# print(my_text.split())


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)

# nltk.download()
# print(wordnet.synsets('watch'))
# Synset

start = time.time()
my_synsets = wordnet.synsets('watch')
print(my_synsets)
# for word in wordnet.synsets('watch', pos='n'):
#     print(word.lemma_names)
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)


# my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# my_list = []
# my_list = [4, 2, 6]
# my_list = [2, 4, 6, 8, 3, 7, 5]
my_list = [2, 3, 4, 5, 6, 7, 8]

new_list = my_list[:]
# for my_element in my_list:
#     new_list.pop(0)
#     print(my_element, new_list)

# build_groups_n(my_list, set())
# print(build_groups_n2(my_list, set()))

# my_group = {2, 4, 6}
# add_group(3, my_group)
# print(my_group)


# dealing with a graph as list of lists
graph = [[0,1,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]]


#function determines the neighbors of a given vertex
def N(vertex):
    c = 0
    l = []
    for i in graph[vertex]:
        if i is 1 :
         l.append(c)
        c+=1

    # print('N', vertex, l)
    return l

#the Bron-Kerbosch recursive algorithm
def bronk(r,p,x):
    if len(p) == 0 and len(x) == 0:
        print r
        return
    for vertex in p[:]:
        r_new = r[::]
        r_new.append(vertex)
        p_new = [val for val in p if val in N(vertex)] # p intersects N(vertex)
        x_new = [val for val in x if val in N(vertex)] # x intersects N(vertex)
        bronk(r_new,p_new,x_new)
        p.remove(vertex)
        x.append(vertex)


# bronk([], [0,1,2,3,4,5], [])


def choose_pivot(list1, list2):

    index = random.randint(0, len(list1) + len(list2) - 1)
    if index < len(list1):
        return list1[index]
    else:
        return list2[index-len(list1)]

def list_difference(list1, list2):
    return [item for item in list1 if item not in list2]



# BronKerbosch2(R,P,X):
#        if P and X are both empty:
#            report R as a maximal clique
#        choose a pivot vertex u in P u X
#        for each vertex v in P \ N(u):
#            BronKerbosch2(R u {v}, P n N(v), X n N(v))
#            P := P \ {v}
#            X := X u {v}

def bronk2(clique, candidates, excluded, clique_list):
    if len(candidates) == 0 and len(excluded) == 0:
        # print clique
        clique_list.append(clique)
        return
    pivot = choose_pivot(candidates, excluded)
    neighbours = get_neighbours(pivot, my_list)
    p_minus_neighbours = list_difference(candidates, neighbours)[:]
    for vertex in p_minus_neighbours:
        vertex_neighbours = get_neighbours(vertex, my_list)
        new_candidates = [val for val in candidates if val in vertex_neighbours]  # p intersects N(vertex)
        new_excluded = [val for val in excluded if val in vertex_neighbours]  # x intersects N(vertex)
        bronk2(clique + [vertex], new_candidates, new_excluded, clique_list)
        candidates.remove(vertex)
        excluded.append(vertex)


def bronk2_synset(clique, candidates, excluded, clique_list, synsets):
    if len(candidates) == 0 and len(excluded) == 0:
        # print clique
        clique_list.append(clique)
        return
    pivot = choose_pivot(candidates, excluded)
    neighbours = get_synset_neigbours(pivot, synsets)
    p_minus_neighbours = list_difference(candidates, neighbours)[:]
    for vertex in p_minus_neighbours:
        vertex_neighbours = get_synset_neigbours(vertex, synsets)
        new_candidates = [val for val in candidates if val in vertex_neighbours]  # p intersects N(vertex)
        new_excluded = [val for val in excluded if val in vertex_neighbours]  # x intersects N(vertex)
        bronk2(clique + [vertex], new_candidates, new_excluded, clique_list)
        candidates.remove(vertex)
        excluded.append(vertex)


def bronk1(r, p, x):
    # print('Before: P', p, 'x', x)

    if len(p) == 0 and len(x) == 0:
        print r
        return
    for vertex in p[:]:
        r_new = r[::]
        r_new.append(vertex)
        p_new = [val for val in p if val in get_neighbours(vertex, my_list)] # p intersects N(vertex)
        x_new = [val for val in x if val in get_neighbours(vertex, my_list)] # x intersects N(vertex)
        bronk1(r_new, p_new, x_new)
        p.remove(vertex)
        x.append(vertex)


def bronker_bosch1(clique, candidates, excluded):
    '''Naive Bron-Kerbosch algorithm'''
    if not candidates and not excluded:
        # print(clique)
        return

    for vertex in list(candidates):
        vertex_neighbours = get_neighbours(vertex, my_list)
        new_candidates = [val for val in candidates if val in vertex_neighbours] # p intersects N(vertex)
        # new_candidates = candidates.intersection(vertex_neighbours)
        new_excluded = [val for val in excluded if val in vertex_neighbours] # p intersects N(vertex)
        # new_excluded = excluded.intersection(vertex_neighbours)
        bronker_bosch1(clique + [vertex], new_candidates, new_excluded)
        candidates.remove(vertex)
        excluded.append(vertex)

# bronker_bosch1([], my_list, [])

# bronk1([], my_list, [])
# aa = []
# bronk2([], my_list, [], aa)
# print(aa)

# bronk2([], my_list, [])
# bronk([], my_list, [])

my_word = 'drinks'
# print(build_groups(my_word))
my_senses = wordnet.synsets(my_word, pos='n')

for sense in my_senses:
    # print(sense)
    for lemma in sense.lemmas:
        print(lemma.name)

