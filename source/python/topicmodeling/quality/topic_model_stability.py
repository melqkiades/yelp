
import logging as log
import random

import numpy
import time

from etl import ETLUtils
from topicmodeling.context import topic_model_creator
from topicmodeling.external.topicensemble.unsupervised import rankings
from topicmodeling.external.topicensemble.unsupervised import util
from topicmodeling.hungarian import HungarianError
from topicmodeling.jaccard_similarity import AverageJaccard, RankingSetAgreement
from utils import utilities
from utils.constants import Constants

TERM_DIFFERENCE = 'term_difference'
TERM_STABILITY_REFERENCE = 'term_stability_reference'
TERM_STABILITY_PAIRWISE = 'term_stability_pairwise'


def evaluate_topic_model(metric):
    print('%s: evaluating topic model' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))

    Constants.update_properties({
        Constants.NUMPY_RANDOM_SEED_FIELD: Constants.NUMPY_RANDOM_SEED + 10,
        Constants.RANDOM_SEED_FIELD: Constants.RANDOM_SEED + 10
    })
    utilities.plant_seeds()
    Constants.print_properties()

    records = ETLUtils.load_json_file(Constants.PROCESSED_RECORDS_FILE)
    if Constants.SEPARATE_TOPIC_MODEL_RECSYS_REVIEWS:
        num_records = len(records)
        records = records[:num_records / 2]
    print('num_reviews', len(records))

    all_term_rankings = None
    topic_model_type = Constants.TOPIC_MODEL_TYPE
    if topic_model_type in ['lda', 'nmf']:
        all_term_rankings = create_all_term_rankings(records, metric)
    elif topic_model_type == 'ensemble':
        all_term_rankings = create_all_term_rankings_from_ensemble()
    else:
        raise ValueError(
            'Unrecognized topic modeling algorithm: \'%s\'' % topic_model_type)
    print('Total iterations: %d' % len(all_term_rankings))

    if metric == TERM_STABILITY_REFERENCE:
        return eval_term_stability_reference(all_term_rankings)
    if metric == TERM_STABILITY_PAIRWISE:
        return eval_term_stability_pairwise(all_term_rankings)
    elif metric == TERM_DIFFERENCE:
        return eval_term_difference(all_term_rankings)
    else:
        raise ValueError('Unknown evaluation metric: \'%s\'' % metric)

        # return eval_term_stability_reference(all_term_rankings)


def create_all_term_rankings(records, metric):
    print('%s: creating all term rankings' % time.strftime("%Y/%m/%d-%H:%M:%S"))

    all_term_rankings = []

    # context_extractor =\
    #     topic_model_creator.create_topic_model(records, None, None)
    # terms_matrix = get_topic_model_terms(
    #     context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
    # all_term_rankings.append(terms_matrix)

    context_extractor = \
        topic_model_creator.train_context_extractor(records, False)
    terms_matrix = get_topic_model_terms(
        context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
    all_term_rankings.append(terms_matrix)

    sample_ratio = Constants.TOPIC_MODEL_STABILITY_SAMPLE_RATIO

    if metric == TERM_STABILITY_PAIRWISE:
        sample_ratio = None
        Constants.update_properties(
            {Constants.TOPIC_MODEL_STABILITY_SAMPLE_RATIO_FIELD: sample_ratio})
        msg = 'Warning: Since the metric is \'%s\' I have updated the ' \
              'topic_model_stability_sample_ratio value to None' % metric
        print(msg)

    num_iterations = Constants.TOPIC_MODEL_STABILITY_ITERATIONS
    for i in range(num_iterations - 1):
        print('Iteration %d/%d' % (i+1, num_iterations))

        if sample_ratio is None:
            sampled_records = records
        else:
            sampled_records = sample_list(records, sample_ratio)
        context_extractor = \
            topic_model_creator.train_context_extractor(sampled_records, False)
        terms_matrix = get_topic_model_terms(
            context_extractor, Constants.TOPIC_MODEL_STABILITY_NUM_TERMS)
        all_term_rankings.append(terms_matrix)

    return all_term_rankings


def create_all_term_rankings_from_ensemble():
    print('%s: creating all term rankings from ensemble' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))

    file_paths = get_topic_ensemble_ranks_file_paths()

    all_term_rankings = []
    top = Constants.TOPIC_MODEL_STABILITY_NUM_TERMS
    for rank_path in file_paths:
        log.debug("Loading term ranking set from %s ..." % rank_path)
        (term_rankings, labels) = util.load_term_rankings(
            rank_path)
        log.debug("Set has %d rankings covering %d terms" % (
            len(term_rankings), rankings.term_rankings_size(term_rankings)))
        # do we need to truncate the number of terms in the ranking?
        if top > 1:
            term_rankings = rankings.truncate_term_rankings(term_rankings,
                                                            top)
            log.debug(
                "Truncated to %d -> set now has %d rankings covering %d terms" % (
                    top, len(term_rankings),
                    rankings.term_rankings_size(term_rankings)))
        all_term_rankings.append(term_rankings)

    return all_term_rankings


def get_topic_ensemble_ranks_file_paths():

    num_models = Constants.TOPIC_MODEL_STABILITY_ITERATIONS
    random_seeds = range(1, num_models + 1)

    suffix = 'ranks_ensemble_k%02d.pkl' % Constants.TOPIC_MODEL_NUM_TOPICS

    file_paths = []

    for seed in random_seeds:
        prefix = 'topic_model_seed-' + str(seed)
        topic_model_folder = Constants.generate_file_name(
            prefix, '', Constants.ENSEMBLE_FOLDER, None, None, True, True)[:-1]
        topic_model_file = topic_model_folder + '/' + suffix
        print(topic_model_file)
        file_paths.append(topic_model_file)

    return file_paths


def sample_list(lst, sample_ratio):

    num_samples = int(len(lst) * sample_ratio)
    sampled_list = [
        lst[i] for i in sorted(random.sample(xrange(len(lst)), num_samples))]

    return sampled_list


def get_topic_model_terms(context_extractor, num_terms):

    context_extractor.num_topics = Constants.TOPIC_MODEL_NUM_TOPICS
    topic_model_strings = context_extractor.print_topic_model(num_terms)
    topic_term_matrix = []

    for topic in range(Constants.TOPIC_MODEL_NUM_TOPICS):
        terms = topic_model_strings[topic].split(" + ")
        terms = [term.partition("*")[2] for term in terms]
        topic_term_matrix.append(terms)

    return topic_term_matrix


# Created by Derek Greene and Mark Belford
def load_term_rankings(file_paths):
    # Load cached ranking sets
    all_term_rankings = []
    for rank_path in file_paths:
        log.debug("Loading test term ranking set from %s ..." % rank_path)
        (term_rankings, labels) = util.load_term_rankings(
            rank_path)
        log.debug("Set has %d rankings covering %d terms" % (
            len(term_rankings), rankings.term_rankings_size(term_rankings)))
        all_term_rankings.append(term_rankings)


# Created by Derek Greene and Mark Belford
def eval_term_stability_reference(all_term_rankings):
    """
    Compares all the topic models against the first topic model of the
    all_term_rankings array and returns the mean stability (as in Greene 2014)

    This function should be used to try to find the ideal number of topics and
    not to measure the stability of a topic modeling algorithm

    :param all_term_rankings: the top terms of the topics of the topic model
    :return: a dictionary with the term stability results
    """

    # First argument was the reference term ranking
    reference_term_ranking = all_term_rankings[0]
    remaining_term_rankings = all_term_rankings[1:]
    r = len(remaining_term_rankings)
    print("Loaded %d non-reference term rankings" % r)

    # Perform the evaluation
    metric = AverageJaccard()
    matcher = RankingSetAgreement(metric)
    print("Performing reference comparisons with %s ..." % str(metric))
    all_scores = []
    for i in range(r):
        try:
            score = \
                matcher.similarity(reference_term_ranking,
                                   remaining_term_rankings[i])
            all_scores.append(score)
        except HungarianError:
            msg = \
                "HungarianError: Unable to find results. Algorithm has failed."
            print(msg)
            all_scores.append(float('nan'))

    # Get overall score across all candidates
    all_scores = numpy.array(all_scores)
    print('Total scores: %d' % len(all_scores))
    print(all_scores)

    # print("Stability=%.4f [%.4f,%.4f]" % (
    #     numpy.nanmean(all_scores), numpy.nanmin(all_scores),
    #     numpy.nanmax(all_scores)))
    #
    # return all_scores

    results = summarize_scores(all_scores, TERM_STABILITY_REFERENCE)

    return results


# Created by Derek Greene and Mark Belford
def eval_term_difference(all_term_rankings):
    # For number of top terms
    top = Constants.TOPIC_MODEL_STABILITY_NUM_TERMS
    log.debug("Comparing unions of top %d terms ..." % top)
    # get the set of all terms used in the top terms for specified model
    all_model_terms = []
    for term_rankings in all_term_rankings:
        model_terms = set()
        for ranking in rankings.truncate_term_rankings(
                term_rankings, top):
            for term in ranking:
                model_terms.add(term)
        all_model_terms.append(model_terms)
    all_scores = []
    # perform pairwise comparisons

    num_models = len(all_term_rankings)
    for i in range(num_models):
        # NB: assume same value of K for both models
        base_k = len(all_term_rankings[i])
        for j in range(i + 1, num_models):
            diff = len(
                all_model_terms[i].symmetric_difference(all_model_terms[j]))
            ndiff = float(diff) / (base_k * top)
            all_scores.append(ndiff)

    # Get overall score across all pairs
    all_scores = numpy.array(all_scores)
    results = summarize_scores(all_scores, TERM_DIFFERENCE)

    return results


# Created by Derek Greene and Mark Belford
def eval_term_stability_pairwise(all_term_rankings):
    """
    Makes a pairwise comparison between each pair of topic models, making in
    total nC2 (n combined 2) comparisons, where n is the number of topic models
    as in (Belford 2018)

    This function should be used to try to measure the stability of a topic
    modeling algorithm and not to find the ideal number of topics

    :param all_term_rankings: the top terms of the topics of the topic model
    :return: a dictionary with the term stability results
    """
    r = len(all_term_rankings)
    metric = rankings.JaccardBinary()
    matcher = rankings.RankingSetAgreement(metric)

    # Perform pairwise comparisons evaluation for all models
    log.info(
        "Evaluating stability %d base term rankings with %s and top %d terms ..." % (
            r, str(metric), Constants.TOPIC_MODEL_STABILITY_NUM_TERMS))
    all_scores = []
    for i in range(r):
        for j in range(i + 1, r):
            try:
                score = matcher.similarity(all_term_rankings[i],
                                           all_term_rankings[j])
                all_scores.append(score)
            except Exception as e:
                log.warning("Error occurred comparing pair (%d,%d): %s" % (
                    i, j, str(e)))
    log.info("Compared %d pairs of term rankings" % len(all_scores))

    # Get overall score across all pairs
    all_scores = numpy.array(all_scores)

    results = summarize_scores(all_scores, TERM_STABILITY_PAIRWISE)

    return results


def summarize_scores(scores, metric_name):
    results = {
        metric_name + '_mean': scores.mean(),
        metric_name + '_median': numpy.median(scores),
        metric_name + '_std': scores.std(),
        metric_name + '_min': scores.min(),
        metric_name + '_max': scores.max(),
    }

    return results


def full_cycle(metric):
    csv_file_name = Constants.generate_file_name(
        metric, 'csv', Constants.RESULTS_FOLDER, None,
        None, False)
    json_file_name = Constants.generate_file_name(
        metric, 'json', Constants.RESULTS_FOLDER, None,
        None, False)
    print(json_file_name)
    print(csv_file_name)

    properties = Constants.get_properties_copy()
    results = evaluate_topic_model(metric)
    print(results)
    results.update(properties)

    ETLUtils.write_row_to_csv(csv_file_name, results)
    ETLUtils.write_row_to_json(json_file_name, results)


def main():

    # metric = TERM_DIFFERENCE
    metric = TERM_STABILITY_PAIRWISE
    # metric = TERM_STABILITY_REFERENCE
    full_cycle(metric)


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
