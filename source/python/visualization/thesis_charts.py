from itertools import product

import pandas
import time

import seaborn
import matplotlib.pyplot as plt

from utils.constants import Constants


# Plots for Section sec:context_richness_results of the thesis
def plot_context_richness_score_pos_types():
    prefix = Constants.RESULTS_FOLDER + Constants.ITEM_TYPE + \
             '_topic_model_context_richness'
    csv_file_path = prefix + '.csv'
    json_file_path = prefix + '.json'

    bow_type_field = 'POS type'
    # bow_type_field = 'bow_type'

    data_frame = pandas.read_csv(csv_file_path)
    data_frame.drop(columns=['cycle_time'], inplace=True)
    data_frame['num_topics'].astype('category')
    data_frame['bow_type'].fillna('All', inplace=True)
    data_frame.rename(columns={'bow_type': bow_type_field}, inplace=True)
    data_frame = data_frame.loc[data_frame[
        Constants.TOPIC_MODEL_TARGET_REVIEWS_FIELD] == Constants.SPECIFIC]
    print(data_frame.describe())
    print(data_frame.head())

    g = seaborn.barplot(
        x='num_topics', y='probability_score', hue=bow_type_field, data=data_frame)
    g.set(xlabel='Number of topics', ylabel='Context richness score')
    plt.ylim(0, 0.14)
    g.figure.savefig(prefix + '_pos_types.pdf')


def plot_context_richness_score_specific_vs_generic():
    prefix = Constants.RESULTS_FOLDER + Constants.ITEM_TYPE + \
             '_topic_model_context_richness'
    csv_file_path = prefix + '.csv'
    json_file_path = prefix + '.json'

    review_type_field = 'Review type'

    data_frame = pandas.read_csv(csv_file_path)
    data_frame.drop(columns=['cycle_time'], inplace=True)
    data_frame['num_topics'].astype('category')
    data_frame['bow_type'].fillna('All', inplace=True)
    data_frame.rename(columns=
        {Constants.TOPIC_MODEL_TARGET_REVIEWS_FIELD: review_type_field}, inplace=True)
    data_frame = data_frame.loc[data_frame['bow_type'] == 'NN']
    print(data_frame.describe())
    print(data_frame.head())

    g = seaborn.barplot(
        x='num_topics', y='probability_score', hue=review_type_field,
        data=data_frame)
    g.set(xlabel='Number of topics', ylabel='Context-richness')
    plt.ylim(0, 0.14)
    g.figure.savefig(prefix + '_specific_vs_generic.pdf')
    # plt.show()


def plot_ats_score():
    metric = 'term_difference'
    metric = 'term_stability_pairwise'

    csv_file_name = Constants.generate_file_name(
        metric, 'csv', Constants.RESULTS_FOLDER, None,
        None, False)
    json_file_name = Constants.generate_file_name(
        metric, 'json', Constants.RESULTS_FOLDER, None,
        None, False)

    data_frame = pandas.read_csv(csv_file_name)
    stability_column = 'term_stability_pairwise_mean'
    topic_model_column = 'Topic modeling algorithm'
    num_topics_field = Constants.TOPIC_MODEL_NUM_TOPICS_FIELD

    data_frame.rename(columns={'topic_model_type': topic_model_column}, inplace=True)
    data_frame[topic_model_column] = data_frame[topic_model_column].map(
        {'lda': 'LDA', 'nmf': 'NMF', 'ensemble': 'Ensemble'})

    g = seaborn.barplot(
        x=num_topics_field, y=stability_column, hue=topic_model_column,
        data=data_frame)
    g.set(xlabel='Number of topics', ylabel='ATS')
    plt.ylim(0, 1.18)
    # g.ylim(10, 40)

    output_folder = Constants.RESULTS_FOLDER + 'pdf/'
    file_name = output_folder + Constants.ITEM_TYPE + '_ats.pdf'
    g.figure.savefig(file_name)

    # plt.show()


def plot_adsd_score():
    metric = 'term_difference'
    # metric = 'term_stability_pairwise'

    csv_file_name = Constants.generate_file_name(
        metric, 'csv', Constants.RESULTS_FOLDER, None,
        None, False)
    json_file_name = Constants.generate_file_name(
        metric, 'json', Constants.RESULTS_FOLDER, None,
        None, False)

    data_frame = pandas.read_csv(csv_file_name)
    stability_column = 'term_difference_mean'
    topic_model_column = 'Topic modeling algorithm'
    num_topics_field = Constants.TOPIC_MODEL_NUM_TOPICS_FIELD

    data_frame.rename(columns={'topic_model_type': topic_model_column}, inplace=True)
    data_frame[topic_model_column] = data_frame[topic_model_column].map(
        {'lda': 'LDA', 'nmf': 'NMF', 'ensemble': 'Ensemble'})

    g = seaborn.barplot(
        x=num_topics_field, y=stability_column, hue=topic_model_column,
        data=data_frame)
    g.set(xlabel='Number of topics', ylabel='ATS')
    plt.ylim(0, 0.7)
    # g.ylim(10, 40)

    output_folder = Constants.RESULTS_FOLDER + 'pdf/'
    file_name = output_folder + Constants.ITEM_TYPE + '_adsd.pdf'
    g.figure.savefig(file_name)

    # plt.show()


def plot_divergence():
    prefix = Constants.RESULTS_FOLDER + Constants.ITEM_TYPE + \
             '_topic_model_divergence'
    csv_file_path = prefix + '.csv'
    json_file_path = prefix + '.json'

    data_frame = pandas.read_csv(csv_file_path)
    # data_frame.drop(columns=['cycle_time'], inplace=True)
    # data_frame['num_topics'].astype('category')
    # data_frame['bow_type'].fillna('All', inplace=True)
    data_frame.rename(columns={'divergence': 'Divergence'}, inplace=True)
    # data_frame = data_frame.loc[data_frame[
    #                                 Constants.TOPIC_MODEL_TARGET_REVIEWS_FIELD] == Constants.SPECIFIC]
    print(data_frame.describe())
    print(data_frame.head())

    # g = seaborn.lineplot(
    #     x='num_topics', y='divergence',
    #     data=data_frame)
    seaborn.set_style("darkgrid")
    data_frame.plot.line(x='num_topics', y='Divergence')
    plt.xlabel('Number of topics')
    plt.ylabel('Divergence')
    # plt.ylim(0, 6)
    # plt.xlim(2, 12)
    # g.figure.savefig(prefix + '.pdf')
    plt.savefig(prefix + '.pdf')
    # plt.show()


def plot_classifier_results():
    csv_file_path = Constants.RESULTS_FOLDER + 'classifier_results.csv'
    data_frame = pandas.read_csv(csv_file_path)

    classifier_column = 'classifier'
    accuracy_column = 'accuracy'
    item_type_column = 'Dataset'

    data_frame.rename(columns={'business_type': item_type_column},
                      inplace=True)
    data_frame[item_type_column] = data_frame[item_type_column].map(
        {'yelp_hotel': 'Yelp Hotel', 'yelp_restaurant': 'Yelp Restaurant'})
    data_frame[classifier_column] = data_frame[classifier_column].map({
        'LogisticRegression': 'Logistic Regression',
        'RandomForestClassifier': 'Random Forest',
        'KNeighborsClassifier': 'K-Neighbours',
        'DummyClassifier': 'Most Frequent',
        'SVC': 'SVC',
        'DecisionTreeClassifier': 'Decision Tree'
    })

    order = [
        'Logistic Regression', 'SVC', 'Random Forest', 'K-Neighbours',
        'Most Frequent', 'Decision Tree'
    ]

    g = seaborn.barplot(
        x=classifier_column, y=accuracy_column, hue=item_type_column,
        data=data_frame, order=order)
    g.set(xlabel='Classifier', ylabel='Accuracy')

    output_folder = Constants.RESULTS_FOLDER + 'pdf/'
    file_name = output_folder + 'classifier_accuracy.pdf'
    plt.ylim(0, 1.0)
    g.figure.savefig(file_name)


def plot_recommender_results(evaluation_set, metric):

    folder = '/tmp/results/'
    file_path = folder + 'rival_%s_results_folds.csv' % Constants.ITEM_TYPE

    data_frame = pandas.read_csv(file_path)

    strategy = {'RMSE': 'user_test', 'Recall@10': 'rel_plus_n'}[metric]

    data_frame = data_frame[data_frame['Strategy'] == strategy]
    data_frame = data_frame[data_frame['Evaluation_Set'] == evaluation_set]
    data_frame.sort_values('Num_Topics', inplace=True)
    context_data_frame = data_frame[data_frame['Context_Format'] == 'context_topic_weights']
    context_data_frame = context_data_frame[['Num_Topics', metric]]
    no_context_data_frame = data_frame[data_frame['Context_Format'] == 'no_context']
    no_context_data_frame = no_context_data_frame[['Num_Topics', metric]]

    if metric == 'RMSE':
        no_context_best_value = no_context_data_frame[metric].min()
    elif metric == 'Recall@10':
        no_context_best_value = no_context_data_frame[metric].max()
    else:
        raise ValueError('Unrecognized metric \'%s\'' % metric)

    data_matrix = context_data_frame.values
    num_topics_data = data_matrix[:, 0]
    context_data = data_matrix[:, 1]
    no_context_data = [no_context_best_value]*len(num_topics_data)

    context_line, = plt.plot(num_topics_data, context_data, label='Rich Context')
    no_context_line, = plt.plot(
        num_topics_data, no_context_data, label='Factorization Machines',
        linestyle='--')

    # font = {'weight': 'bold',
    #         'size': 22}

    # matplotlib.rc('font', **font)

    plt.legend(handles=[context_line, no_context_line], prop={'size': 14})
    # plt.legend(handles=[context_line, no_context_line])


    if metric == 'RMSE':
        plt.ylim(0.9, 1.4)
    if metric == 'Recall@10':
        if Constants.ITEM_TYPE == 'yelp_hotel':
            plt.ylim(0.0, 0.45)
        else:
            plt.ylim(0.0, 0.1)
    plt.xlabel('Number of topics', fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.xlabel('Number of topics')
    plt.ylabel(metric)

    # divergence = {'yelp_hotel': 9, 'yelp_restaurant': 34}[Constants.ITEM_TYPE]
    # plt.axvline(x=divergence)
    export_folder = '/tmp/'
    file_path = export_folder + 'plot_%s_%s_%s.pdf' % (
        metric, Constants.ITEM_TYPE, evaluation_set)
    file_path = file_path.lower()
    plt.savefig(file_path, figsize=(4, 4), bbox_inches='tight')
    plt.cla()
    plt.clf()


    # seaborn.set_style("darkgrid")
    # data_frame.plot.line(x='Num_Topics', y='RMSE', hue='Context_Format')
    # plt.xlabel('Number of topics')
    # plt.ylabel('Divergence')
    # plt.ylim(0.9, 1.3)
    #
    # plt.savefig('/tmp/fourcity_hotel_rmse.pdf')


def main():
    print('%s: Started running the thesis charts module' %
          time.strftime("%Y/%m/%d-%H:%M:%S"))
    # plot_context_richness_score_pos_types()
    # plot_context_richness_score_specific_vs_generic()
    # plot_ats_score()
    # plot_adsd_score()
    # plot_divergence()
    # plot_classifier_results()
    evaluation_sets = ['test_users', 'test_only_users']
    metrics = ['RMSE', 'Recall@10']
    for evaluation_set, metric in product(evaluation_sets, metrics):
        plot_recommender_results(evaluation_set, metric)
    # plot_recommender_results('test_only_users')

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)


