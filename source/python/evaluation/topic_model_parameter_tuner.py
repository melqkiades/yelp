import time

from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe
from hyperopt.mongoexp import MongoTrials


def run_recommender(args):
    import sys
    # sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')
    sys.path.append('/home/fpena/yelp/source/python')
    from utils.constants import Constants
    from topicmodeling.context import topic_model_analyzer

    print('\n\n************************\n************************\n')
    print('args', args)

    parameters = {
        'business_type': args['business_type'],
        # 'lda_alpha': args['lda_alpha'],
        # 'lda_beta': args['lda_beta'],
        'lda_epsilon': args['lda_epsilon'],
        'lda_model_iterations': int(args['lda_model_iterations']),
        'lda_model_passes': int(args['lda_model_passes']),
        'lda_num_topics': int(args['lda_num_topics']),
        # 'topic_weighting_method': args['topic_weighting_method'],
        'use_context': args['use_context']
    }

    Constants.update_properties(parameters)
    # Finish updating parameters

    results = topic_model_analyzer.export_topics(0, 0)
    results['loss'] = -results['combined_score']
    results['status'] = 'ok'

    print('loss', results['loss'])

    return results


def tune_parameters():

    from utils.constants import Constants

    context_name = '_context' if Constants.USE_CONTEXT else '_nocontext'

    mongo_url =\
        'mongo://localhost:1234/topicmodel_' +\
        Constants.ITEM_TYPE + context_name + '/jobs'
    trials = MongoTrials(mongo_url, exp_key='exp1')

    print('Connected to %s' % mongo_url)

    space =\
        hp.choice('use_context', [
            {
                'business_type': Constants.ITEM_TYPE,
                # 'lda_alpha': hp.uniform('lda_alpha', 0, 1),
                # 'lda_beta': hp.uniform('lda_beta', 0, 2),
                'lda_epsilon': hp.uniform('lda_epsilon', 0, 0.5),
                'lda_model_iterations': hp.quniform('lda_model_iterations', 50, 500, 1),
                'lda_model_passes': hp.quniform('lda_model_passes', 1, 100, 1),
                'lda_num_topics': hp.quniform('lda_num_topics', 1, 1000, 1),
                # 'topic_weighting_method': hp.choice('topic_weighting_method', ['probability', 'binary', 'all_topics']),
                'use_context': True
            },
        ])

    best = fmin(
        run_recommender, space=space, algo=tpe.suggest,
        max_evals=1000, trials=trials)

    print('losses', sorted(trials.losses()))
    print(
        'best', trials.best_trial['result'], trials.best_trial['misc']['vals'])
    print('num trials: %d' % len(trials.losses()))


start = time.time()
tune_parameters()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)