import os
import time
import dill as pickle

from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe
from hyperopt.mongoexp import MongoTrials

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt


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

    if Constants.USE_CONTEXT:
        parameters = {
            'business_type': args['business_type'],
            # 'fm_init_stdev': args['fm_init_stdev'],
            'fm_iterations': int(args['fm_iterations']),
            'fm_num_factors': int(args['fm_num_factors']),
            'fm_use_1way_interactions': args['fm_use_1way_interactions'],
            'fm_use_bias': args['fm_use_bias'],
            # 'lda_alpha': args['lda_alpha'],
            # 'lda_beta': args['lda_beta'],
            'lda_epsilon': args['lda_epsilon'],
            'lda_model_iterations': int(args['lda_model_iterations']),
            # 'lda_model_passes': int(args['lda_model_passes']),
            'lda_num_topics': int(args['lda_num_topics']),
            # 'topic_weighting_method': args['topic_weighting_method'],
            # 'use_no_context_topics_sum': args['use_no_context_topics_sum'],
            'use_context': args['use_context']
        }
    else:
        parameters = {
            'business_type': args['business_type'],
            # 'fm_init_stdev': args['fm_init_stdev'],
            'fm_iterations': int(args['fm_iterations']),
            'fm_num_factors': int(args['fm_num_factors']),
            'fm_use_1way_interactions': args['fm_use_1way_interactions'],
            'fm_use_bias': args['fm_use_bias'],
            'use_context': args['use_context']
        }

    Constants.update_properties(parameters)
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

    space =\
        hp.choice('use_context', [
            {
                'business_type': Constants.ITEM_TYPE,
                # 'fm_init_stdev': hp.uniform('fm_init_stdev', 0, 2),
                'fm_iterations': hp.quniform('fm_context_iterations', 100, 500, 1),
                'fm_num_factors': hp.quniform('fm_context_num_factors', 0, 50, 1),
                'fm_use_1way_interactions': hp.choice('fm_use_1way_interactions', [True, False]),
                'fm_use_bias': hp.choice('use_bias', [True, False]),
                # 'lda_alpha': hp.uniform('lda_alpha', 0, 1),
                # 'lda_beta': hp.uniform('lda_beta', 0, 2),
                'lda_epsilon': hp.uniform('lda_epsilon', 0, 0.5),
                'lda_model_iterations': hp.quniform('lda_model_iterations', 50, 500, 1),
                # 'lda_model_passes': hp.quniform('lda_model_passes', 1, 10, 1),
                'lda_num_topics': hp.quniform('lda_num_topics', 1, 300, 1),
                # 'topic_weighting_method': hp.choice('topic_weighting_method', ['probability', 'binary', 'all_topics']),
                # 'use_no_context_topics_sum': hp.choice('use_no_context_topics_sum', [True, False]),
                'use_context': True
            },
        ])

    if not Constants.USE_CONTEXT:
        space = \
            hp.choice('no_context', [
                {
                    'business_type': Constants.ITEM_TYPE,
                    # 'fm_init_stdev': hp.uniform('fm_init_stdev', 0, 2),
                    'fm_iterations': hp.quniform('fm_context_iterations', 100,
                                                 500, 1),
                    'fm_num_factors': hp.quniform('fm_context_num_factors', 0,
                                                  50, 1),
                    'fm_use_1way_interactions': hp.choice(
                        'fm_use_1way_interactions', [True, False]),
                    'fm_use_bias': hp.choice('use_bias', [True, False]),
                    'use_context': False
                },
            ])

    best = fmin(
        run_recommender, space=space, algo=tpe.suggest,
        max_evals=100, trials=trials)

    # print('\n\n')
    #
    # for trial in trials:
    #     # print(trial)
    #     print(trial['misc']['vals'], trial['result']['loss'])
    print('losses', trials.losses())
    print(
        'best', trials.best_trial['result'], trials.best_trial['misc']['vals'])
    #
    # trials_path = os.path.expanduser('~/tmp/trials-context-2.pkl')
    # with open(trials_path, 'wb') as write_file:
    #     pickle.dump(trials, write_file, pickle.HIGHEST_PROTOCOL)


start = time.time()
tune_parameters()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
