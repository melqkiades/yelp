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
        Constants.BUSINESS_TYPE_FIELD: args[Constants.BUSINESS_TYPE_FIELD],
        # 'lda_alpha': args['lda_alpha'],
        # 'lda_beta': args['lda_beta'],
        Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD:
            args[Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD],
        Constants.TOPIC_MODEL_ITERATIONS_FIELD:
            int(args[Constants.TOPIC_MODEL_ITERATIONS_FIELD]),
        Constants.TOPIC_MODEL_PASSES_FIELD:
            int(args[Constants.TOPIC_MODEL_PASSES_FIELD]),
        Constants.TOPIC_MODEL_NUM_TOPICS_FIELD:
            int(args[Constants.TOPIC_MODEL_NUM_TOPICS_FIELD]),
        # 'topic_weighting_method': args['topic_weighting_method'],
        Constants.USE_CONTEXT_FIELD: args[Constants.USE_CONTEXT_FIELD]
    }

    Constants.update_properties(parameters)
    # Finish updating parameters

    results = topic_model_analyzer.export_topics()
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
        hp.choice(Constants.USE_CONTEXT_FIELD, [
            {
                Constants.BUSINESS_TYPE_FIELD: Constants.ITEM_TYPE,
                # 'lda_alpha': hp.uniform('lda_alpha', 0, 1),
                # 'lda_beta': hp.uniform('lda_beta', 0, 2),
                Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD: hp.uniform(
                    Constants.CONTEXT_EXTRACTOR_EPSILON_FIELD, 0, 0.5),
                Constants.TOPIC_MODEL_ITERATIONS_FIELD: hp.quniform(
                    Constants.TOPIC_MODEL_ITERATIONS_FIELD, 50, 500, 1),
                Constants.TOPIC_MODEL_PASSES_FIELD: hp.quniform(
                    Constants.TOPIC_MODEL_PASSES_FIELD, 1, 100, 1),
                Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: hp.quniform(
                    Constants.TOPIC_MODEL_NUM_TOPICS_FIELD, 1, 1000, 1),
                # 'topic_weighting_method': hp.choice(
                #     'topic_weighting_method',
                #     ['probability', 'binary', 'all_topics']),
                Constants.USE_CONTEXT_FIELD: True
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
