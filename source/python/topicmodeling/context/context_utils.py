from collections import Counter
import cPickle as pickle
import json
import math
import random
import itertools

import h5py
import networkx
import sklearn
from networkx.algorithms.approximation import dominating_set
from networkx.algorithms.approximation import vertex_cover
import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset
import numpy as np
import time
import scipy.misc
from sklearn.cluster import AffinityPropagation, DBSCAN

from datamining import cluster_evaluation
from datamining.cluster_evaluation import DunnCalculator
from topicmodeling.context.sense_clusterer import BaumanSensesGrouper
from topicmodeling.context.senses_group import SenseGroup
from topicmodeling.context.review import Review
from utils.constants import Constants

__author__ = 'fpena'

from nltk import tokenize


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
            if sense.name() not in sense_word_map:
                sense_word_map[sense.name()] = []
            sense_word_map[sense.name()].append(noun)

    all_senses = list(all_senses)
    all_senses_names = [sense.name() for sense in all_senses]

    print('number of senses:', len(all_senses))
    senses_similarity_matrix = build_sense_similarity_matrix(all_senses)

    groups = []
    bronk2_synset([], all_senses_names, [], groups, senses_similarity_matrix)
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
        similarity_matrix[sense.name()] = {}

    index = 1
    num_senses = len(senses)
    for sense1 in senses:
        similarity_matrix[sense1.name()][sense1.name()] = 1.0
        for sense2 in senses[index:]:
            similarity = sense1.wup_similarity(sense2)
            similarity_matrix[sense1.name()][sense2.name()] = similarity
            similarity_matrix[sense2.name()][sense1.name()] = similarity

        if not index % 100:
            print('%s: completed %d/%d senses' %
                  (time.strftime("%Y/%d/%m-%H:%M:%S"), index, num_senses))
        index += 1

    print('finished senses similarity matrix', time.strftime("%H:%M:%S"))

    return similarity_matrix


def get_synset_neighbours(synset, similarity_matrix):
    neighbours = []

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

    :type reviews: list[dict]
    :param reviews:
    """
    text_reviews = []
    for review in reviews:
        text_reviews.append(review[Constants.TEXT_FIELD])

    return text_reviews


def calculate_group_weighted_frequency(group, reviews):
    num_reviews = 0.0

    for review in reviews:
        review_senses = set([sense.name() for sense in review.senses])
        if not frozenset(review_senses).isdisjoint(frozenset(group.senses)):
            num_reviews += 1

    return num_reviews / len(reviews)


def get_context_similarity(context1, context2, topic_indices):

    # We filter the topic model, selecting only the topics that contain context
    filtered_context1 = np.array([context1[i[0]] for i in topic_indices])
    filtered_context2 = np.array([context2[i[0]] for i in topic_indices])

    return 1 / (1 + np.linalg.norm(filtered_context1-filtered_context2))


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


def create_graph(reviews_file, graph_file):

    print('%s: start' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    with open(reviews_file, 'rb') as read_file:
        reviews = pickle.load(read_file)
    reviews = reviews[:13]

    print('num reviews: %d' % len(reviews))
    print('%s: loaded reviews' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    all_nouns = list(get_all_nouns(reviews))
    print('num nouns: %d' % len(all_nouns))
    print('%s: obtained nouns' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    all_senses = generate_all_senses(reviews)
    total_possible_vertices = scipy.misc.comb(len(all_senses), 2, exact=True)
    print('num senses: %d' % len(all_senses))
    print('total possible senses: %d' % total_possible_vertices)
    print('%s: obtained senses' % time.strftime("%Y/%d/%m-%H:%M:%S"))

    graph = networkx.Graph()

    for sense in all_senses:
        graph.add_node(sense.name())
    print('%s: created graph' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    print('num nodes: %d' % len(graph.nodes()))

    cycle = 0
    for synset1, synset2 in itertools.combinations(all_senses, 2):
        cycle += 1
        if not cycle % 100000:
            print('sense cycle: %d/%d\r' % (cycle, total_possible_vertices)),
        if synset1.wup_similarity(synset2) >= 0.7:
            graph.add_edge(synset1.name(), synset2.name())
    print('%s: added vertices' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    print('num edges: %d' % len(graph.edges()))

    with open(graph_file, 'wb') as write_file:
        pickle.dump(graph, write_file, pickle.HIGHEST_PROTOCOL)

    with open(graph_file, 'rb') as read_file:
        graph = pickle.load(read_file)

    print('num nodes: %d' % len(graph.nodes()))
    print('num edges: %d' % len(graph.edges()))

    my_dominating_set = dominating_set.min_weighted_dominating_set(graph)
    print('%s found dominating set' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    print('dominating set length: %d' % len(my_dominating_set))
    my_vertex_cover = vertex_cover.min_weighted_vertex_cover(graph)
    print('%s found vertex cover' % time.strftime("%Y/%d/%m-%H:%M:%S"))
    print('vertex cover length: %d' % len(my_vertex_cover))


def build_hdf5_sense_similarity_matrix(senses):
    """

    :type senses: list[Synset]
    :param senses:
    """

    num_senses = len(senses)

    hdf5_file = Constants.DATASET_FOLDER + Constants.ITEM_TYPE +\
                '_sense_similarity_matrix.hdf5'
    f = h5py.File(hdf5_file, 'w')
    similarity_matrix = f.create_dataset(
        Constants.ITEM_TYPE + "_sense_similarity_matrix",
        (num_senses, num_senses)
    )

    sense_index_map = {}

    sense_index = 0
    for sense in senses:

        if sense in sense_index_map:
            raise ValueError('There are repeated items in the senses iterable')

        sense_index_map[sense.name()] = sense_index
        sense_index += 1

    sense_index_map_file = Constants.DATASET_FOLDER + Constants.ITEM_TYPE +\
        '_sense_index_map.pkl'
    with open(sense_index_map_file, 'wb') as write_file:
        pickle.dump(sense_index_map, write_file, pickle.HIGHEST_PROTOCOL)

    print('building senses similarity matrix', time.strftime("%H:%M:%S"))

    index = 1
    for sense1 in senses:
        sense1_index = sense_index_map[sense1.name()]
        similarity_matrix[sense1_index, sense1_index] = 1.0
        for sense2 in senses[index:]:
            similarity = sense1.wup_similarity(sense2)
            sense2_index = sense_index_map[sense2.name()]
            similarity_matrix[sense1_index, sense2_index] = similarity
            similarity_matrix[sense2_index, sense1_index] = similarity

        if not index % 100:
            print('%s: completed %d/%d senses' %
                  (time.strftime("%Y/%d/%m-%H:%M:%S"), index, num_senses))
        index += 1

    print('finished senses similarity matrix', time.strftime("%H:%M:%S"))

    print(similarity_matrix[1, 1])
    print(f[Constants.ITEM_TYPE + "_sense_similarity_matrix"][0][0])
    # print(similarity_matrix)

    f.close()


def main():

    base_dir = Constants.DATASET_FOLDER
    dataset = Constants.ITEM_TYPE
    print('Building %s similarity matrix' % Constants.ITEM_TYPE)

    reviews_file = base_dir + 'reviews_' + dataset + '_shuffled.pkl'

    with open(reviews_file, 'rb') as read_file:
        reviews = pickle.load(read_file)
        # reviews = reviews[20:30]

    all_senses = list(generate_all_senses(reviews))
    print('num senses: %d' % len(all_senses))
    build_hdf5_sense_similarity_matrix(all_senses)


def build_groups2(nouns):

    print('building groups', time.strftime("%H:%M:%S"))
    all_senses = set()

    sense_word_map = {}
    for noun in nouns:
        senses = wordnet.synsets(noun, pos='n')
        all_senses.update(senses)
        for sense in senses:
            if sense.name() not in sense_word_map:
                sense_word_map[sense.name()] = []
            sense_word_map[sense.name()].append(noun)

    all_senses = list(all_senses)
    all_senses_names = [sense.name() for sense in all_senses]

    print('number of senses:', len(all_senses))
    sense_similarity_matrix, sense_similarity_matrix_columns =\
        get_sense_similarity_submatrix(all_senses_names)
    print('submatrix ready', time.strftime("%H:%M:%S"))

    # affinity_propagation = AffinityPropagation()
    # labels1 = affinity_propagation.fit_predict(sense_similarity_matrix)
    # print('affinity propagation ready', time.strftime("%H:%M:%S"))

    grouper = BaumanSensesGrouper(sense_similarity_matrix, 0.7)
    groups = grouper.group_senses()
    print('groups')
    # print(groups)
    new_groups = []
    for group in groups:
        new_group = set()
        for element in group:
            sense_name = sense_similarity_matrix_columns[element]
            new_group.add(sense_name)
        new_groups.append(new_group)

    print('finished groups', time.strftime("%H:%M:%S"))
    # print(groups)
    # print(new_groups)
    print('num groups: %d' % len(groups))

    sense_groups = []
    for group in new_groups:
        sense_group = SenseGroup(group)
        for sense in sense_group.senses:
            sense_group.nouns |= set(sense_word_map[sense])
        sense_groups.append(sense_group)

    return sense_groups


def get_sense_similarity_submatrix(sense_names_list):

    columns = []
    sense_index_map_file = Constants.DATASET_FOLDER + Constants.ITEM_TYPE +\
        '_sense_index_map.pkl'
    with open(sense_index_map_file, 'rb') as read_file:
        sense_index_map = pickle.load(read_file)

    inverse_sense_index_map = {v: k for k, v in sense_index_map.iteritems()}

    for sense_name in sense_names_list:
        columns.append(sense_index_map[sense_name])

    columns.sort()
    submatrix_senses = [inverse_sense_index_map[column] for column in columns]

    hdf5_file = Constants.DATASET_FOLDER + Constants.ITEM_TYPE + '_sense_similarity_matrix.hdf5'
    f = h5py.File(hdf5_file, 'r')
    similarity_matrix = f[Constants.ITEM_TYPE + "_sense_similarity_matrix"]
    matrix_size = len(similarity_matrix)
    print('sense similarity matrix length: %d' % matrix_size)
    # print(similarity_matrix[1, 1])

    # columns = range(5)
    submatrix = similarity_matrix[columns, :][:, columns]
    print('%s: obtained submatrix, length: %d' % (time.strftime("%Y/%d/%m-%H:%M:%S"), len(submatrix)))
    # new_senses = []

    return submatrix, submatrix_senses


def evaluate_clustering():

    similarity_matrix = get_sense_similarity_submatrix(range(10000))
    matrix_size = len(similarity_matrix)
    print('got matrix')

    affinity_propagation = AffinityPropagation()
    labels1 = affinity_propagation.fit_predict(similarity_matrix)
    print('affinity propagation')

    dbscan = DBSCAN(min_samples=1)
    labels2 = dbscan.fit_predict(similarity_matrix)
    print('print dbscan')

    distance_matrix = np.ndarray((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            distance_matrix[i, j] = 1 - similarity_matrix[i, j]

    print(distance_matrix[1, 2])
    print(distance_matrix[1, 1])

    print('created distance matrix')

    cluster_map1 = cluster_evaluation.fpena_get_clusters(labels1)
    cluster_map2 = cluster_evaluation.fpena_get_clusters(labels2)

    print(cluster_map1)
    print(cluster_map2)

    sc1 = sklearn.metrics.silhouette_score(distance_matrix, labels1, metric='euclidean')
    sc2 = sklearn.metrics.silhouette_score(distance_matrix, labels2, metric='euclidean')
    sc5 = cluster_evaluation.fpena_evaluate(cluster_map1, distance_matrix)
    sc6 = cluster_evaluation.fpena_evaluate(cluster_map2, distance_matrix)

    num_elements1 = [len(values) for values in cluster_map1.values()]
    num_elements2 = [len(values) for values in cluster_map2.values()]
    print(num_elements1)
    print(num_elements2)

    print('Number of clusters Affinity Propagation: %f' % len(cluster_map1))
    print('Number of clusters DBSCAN: %f' % len(cluster_map2))
    print('Average elements per cluster Affinity Propagation: %f' % np.mean(num_elements1))
    print('Average elements per cluster DBSCAN: %f' % np.mean(num_elements2))
    print('Standard deviation per cluster Affinity Propagation: %f' % np.std(num_elements1))
    print('Standard deviation per cluster DBSCAN: %f' % np.std(num_elements2))
    print('Silouhette score Affinity Propagation (distance matrix): %f' % sc1)
    print('Silouhette score DBSCAN (distance matrix): %f' % sc2)
    print('Dunn index Affinity Propagation (distance matrix): %f' % sc5)
    print('Dunn index DBSCAN (distance matrix): %f' % sc6)


# start = time.time()
# main()
# evaluate_clustering()
# get_similarity_submatrix()
# end = time.time()
# total_time = end - start
# print("Total time = %f seconds" % total_time)
