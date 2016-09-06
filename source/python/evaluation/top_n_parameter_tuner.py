import time

from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe
from hyperopt.mongoexp import MongoTrials


def fibonacci(n):

    a = 1
    b = 1

    sequence = [a, b]

    if n < 2:
        return 1

    for i in range(2, n):
        b_temp = b
        b += a
        a = b_temp
        sequence.append(b)

    return sequence


def run_recommender(args):
    import sys
    # sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')
    sys.path.append('/home/fpena/yelp/source/python')
    from utils.constants import Constants
    from evaluation.context_top_n_runner import ContextTopNRunner

    print('\n\n************************\n************************\n')
    print('args', args)

    # Cast integer values
    # args['fm_iterations'] = int(args['fm_iterations'])
    args[Constants.FM_NUM_FACTORS_FIELD] = \
        int(args[Constants.FM_NUM_FACTORS_FIELD])
    if args[Constants.USE_CONTEXT_FIELD]:
        args[Constants.TOPIC_MODEL_ITERATIONS_FIELD] = \
            int(args[Constants.TOPIC_MODEL_ITERATIONS_FIELD])
        args[Constants.TOPIC_MODEL_PASSES_FIELD] = \
            int(args[Constants.TOPIC_MODEL_PASSES_FIELD])
        args[Constants.TOPIC_MODEL_NUM_TOPICS_FIELD] = \
            int(args[Constants.TOPIC_MODEL_NUM_TOPICS_FIELD])

    Constants.update_properties(args)

    # Finish updating parameters

    my_context_top_n_runner = ContextTopNRunner()
    results = my_context_top_n_runner.run()
    results['loss'] = -results[Constants.EVALUATION_METRIC]
    results['status'] = 'ok'

    print('loss', results['loss'])

    return results


def tune_parameters():

    # trials = Trials()
    from utils.constants import Constants

    context_name = '_context' if Constants.USE_CONTEXT else '_nocontext'
    cycle = '_' + str(Constants.NESTED_CROSS_VALIDATION_CYCLE)

    mongo_url =\
        'mongo://localhost:1234/' +\
        Constants.ITEM_TYPE + context_name + '_db_nested' + cycle + '/jobs'
    trials = MongoTrials(mongo_url, exp_key='exp1')

    print('Connected to %s' % mongo_url)

    params = Constants.get_properties_copy()
    params.update({
        Constants.BUSINESS_TYPE_FIELD: Constants.ITEM_TYPE,
        Constants.TOPN_NUM_ITEMS_FIELD: Constants.TOPN_NUM_ITEMS,
        Constants.NESTED_CROSS_VALIDATION_CYCLE_FIELD:
            Constants.NESTED_CROSS_VALIDATION_CYCLE,
        # 'fm_init_stdev': hp.uniform('fm_init_stdev', 0, 2),
        # 'fm_iterations': hp.quniform('fm_context_iterations', 100, 500, 1),
        Constants.FM_NUM_FACTORS_FIELD: hp.quniform(
            Constants.FM_NUM_FACTORS_FIELD, 0, 200, 1),
        # 'fm_use_1way_interactions': hp.choice('fm_use_1way_interactions', [True, False]),
        # 'fm_use_bias': hp.choice('use_bias', [True, False]),
        # 'lda_alpha': hp.uniform('lda_alpha', 0, 1),
        # 'lda_beta': hp.uniform('lda_beta', 0, 2),
        # Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD: hp.uniform(
        #     Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD, 0, 0.5),
        Constants.TOPIC_MODEL_ITERATIONS_FIELD: hp.quniform(
            Constants.TOPIC_MODEL_ITERATIONS_FIELD, 50, 500, 1),
        Constants.TOPIC_MODEL_PASSES_FIELD: hp.quniform(
            Constants.TOPIC_MODEL_PASSES_FIELD, 1, 100, 1),
        # Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: hp.quniform(
        #     Constants.TOPIC_MODEL_NUM_TOPICS_FIELD, 1, 1000, 1),
        Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: hp.choice(
            Constants.TOPIC_MODEL_NUM_TOPICS_FIELD,
            [10, 20, 30, 50, 75, 100, 150, 300]),
        Constants.TOPIC_MODEL_TYPE_FIELD: hp.choice(
            Constants.TOPIC_MODEL_TYPE_FIELD, ['lda', 'mnf']),
        # 'topic_weighting_method': hp.choice(
        #     'topic_weighting_method',
        #     ['probability', 'binary', 'all_topics']),
        # 'use_no_context_topics_sum': hp.choice(
        #     'use_no_context_topics_sum', [True, False]),
        Constants.USE_CONTEXT_FIELD: Constants.USE_CONTEXT
    })
    params = Constants.get_properties_copy()

    space =\
        hp.choice(Constants.USE_CONTEXT_FIELD, [
            params,
        ])

    if not Constants.USE_CONTEXT:
        unwanted_args = [
            Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD,
            Constants.TOPIC_MODEL_ITERATIONS_FIELD,
            Constants.TOPIC_MODEL_PASSES_FIELD,
            Constants.TOPIC_MODEL_NUM_TOPICS_FIELD
        ]

        for element in space.pos_args[1].named_args[:]:
            if element[0] in unwanted_args:
                space.pos_args[1].named_args.remove(element)

    # best = fmin(
    #     run_recommender, space=space, algo=tpe.suggest,
    #     max_evals=100, trials=trials)

    print('losses', sorted(trials.losses()))
    print(
        'best', trials.best_trial['result']['loss'],
        trials.best_trial['misc']['vals'])
    print('num trials: %d' % len(trials.losses()))


start = time.time()
tune_parameters()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
