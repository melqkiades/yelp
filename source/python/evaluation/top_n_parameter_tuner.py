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

    use_context = args['use_context']

    parameters = {
        'fm_num_factors': args['fm_num_factors'],
        'fm_iterations': args['fm_iterations'],
        'use_context': use_context
    }

    if use_context:
        parameters.update({'lda_num_topics': args['lda_num_topics']})

    Constants.update_properties(parameters)
    # Finish updating parameters

    my_context_top_n_runner = ContextTopNRunner()
    results = my_context_top_n_runner.perform_cross_validation()
    results['loss'] = -results[Constants.EVALUATION_METRIC]
    results['status'] = 'ok'

    print('loss', results['loss'])

    return results


def tune_parameters():

    # trials = Trials()
    from utils.constants import Constants
    mongo_url =\
        'mongo://localhost:1234/' + Constants.ITEM_TYPE + '_context_db/jobs'
    trials = MongoTrials(mongo_url, exp_key='exp1')

    # space = \
        # [
        # hp.choice('nocontext_num_factors', fibonacci(13)[1:]),
        # hp.quniform('nocontext_iterations', 50, 500, 50)
    # ]

    space =\
        hp.choice('use_context', [
            {
                'use_context': False,
                'fm_num_factors': hp.choice('nocontext_num_factors', fibonacci(13)[1:]),
                'fm_iterations': hp.quniform('nocontext_iterations', 50, 500, 50)
            },
            # {
            #     'use_context': True,
            #     'fm_num_factors': hp.choice('context_num_factors', fibonacci(13)[1:]),
            #     'fm_iterations': hp.quniform('context_iterations', 50, 500, 50),
            #     'lda_num_topics': hp.choice('lda_num_topics', [30, 50, 75, 100, 150, 300])
            # },
        ])

    best = fmin(
        run_recommender, space=space, algo=tpe.suggest,
        max_evals=100, trials=trials)

    # print('\n\n')
    #
    # for trial in trials:
    #     # print(trial)
    #     print(trial['misc']['vals'], trial['result']['loss'])
    print('best', best, min(trials.losses()))
    #
    # trials_path = os.path.expanduser('~/tmp/trials-context-2.pkl')
    # with open(trials_path, 'wb') as write_file:
    #     pickle.dump(trials, write_file, pickle.HIGHEST_PROTOCOL)


def plot_trials():
    trials_path = os.path.expanduser('~/tmp/trials.pkl')
    with open(trials_path, 'rb') as read_file:
        trials = pickle.load(read_file)

    f, ax = plt.subplots(1)

    num_factors_list = []
    num_iterations_list = []
    ys = []
    metric_name = 'recall'
    for trial in trials.trials:
        values = trial['misc']['vals']
        print('values', values)

        metric = -trial['result']['loss']
        # print('num_factors', num_factors)
        # print('num_iterations', num_iterations)
        # print('loss', loss)
        num_factors = values['fm_num_factors'][0]
        num_iterations = values['fm_iterations'][0]
        num_factors_list.append(num_factors)
        num_iterations_list.append(num_iterations)
        ys.append(metric)

    normalized_ys = [float(i) / max(ys) for i in ys]

    ax.scatter(num_factors_list, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('No context $' + metric_name + '$ $vs$ $factors$ ', fontsize=18)
    ax.set_xlabel('$factors$', fontsize=16)
    ax.set_ylabel('$' + metric_name + '$', fontsize=16)

    num_factors_figure = os.path.expanduser('~/tmp/trials_num_factors.pdf')
    # plt.show()
    # plt.savefig(num_factors_figure)

    f, ax = plt.subplots(1)
    ax.scatter(num_iterations_list, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('No context $' + metric_name + '$ $vs$ $iterations$ ', fontsize=18)
    ax.set_xlabel('$iterations$', fontsize=16)
    ax.set_ylabel('$' + metric_name + '$', fontsize=16)

    num_iterations_figure = os.path.expanduser('~/tmp/trials_num_iterations.pdf')

    f, ax = plt.subplots(1)
    ax.scatter(num_iterations_list, num_factors_list, c=normalized_ys, cmap='plasma', s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('No context $' + metric_name + '$ $vs$ $iterations$ ', fontsize=18)
    ax.set_xlabel('$iterations$', fontsize=16)
    ax.set_ylabel('$' + metric_name + '$', fontsize=16)

    print('best', trials.best_trial['misc']['vals'], min(trials.losses()))

    plt.show()

start = time.time()
tune_parameters()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
