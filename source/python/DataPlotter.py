__author__ = 'franpena'

import json
from pandas import DataFrame
import matplotlib.pyplot as plt


class DataPlotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_json_file(file_path, column, plot_type='line', title=None,
                       x_label=None, y_label=None, show_total=True,
                       show_range=False, x_scale='linear', y_scale='linear'):
        """
        Creates a DataFrame object from a JSON file and returns the plot of the
        data including the values for the mean, median, standard deviation, and
        if requested, the sum of all the values, and a range with the minimum
        and maximum values

        @param file_path: the absolute path of the JSON file that contains the
        data
        @param column: the column which will be used to group and count the data
        @param plot_type: the type of graph. For example 'bar', 'barh', 'line',
        etc.
        @param title: the title of the graph
        @param x_label: the label for the x axis
        @param y_label: the label for the y axis
        @param show_total: a boolean which indicates if the sum of all the
        values should be displayed on the graph
        @param show_range: a boolean which indicates if the minimum and maximum
        values should be displayed on the graph
        """
        records = [json.loads(line) for line in open(file_path)]

        # Inserting all records stored in form of lists in to 'pandas DataFrame'
        data_frame = DataFrame(records)
        return DataPlotter.plot_data(data_frame, column, plot_type, title,
                                     x_label, y_label, show_total, show_range,
                                     x_scale, y_scale)

    @staticmethod
    def plot_data(data_frame, column, plot_type='line', title=None,
                  x_label=None, y_label=None, show_total=True,
                  show_range=False, x_scale='linear', y_scale='linear'):
        """
        Returns the plot of the DataFrame object including the values for the
        mean, median, standard deviation, and if requested, the sum of all the
        values, and a range with the minimum and maximum values

        @param data_frame: the DataFrame object that contains the data to be
        plotted
        @param column: the column which will be used to group and count the data
        @param plot_type: the type of graph. For example 'bar', 'barh', 'line',
        etc.
        @param title: the title of the graph
        @param x_label: the label for the x axis
        @param y_label: the label for the y axis
        @param show_total: a boolean which indicates if the sum of all the
        values should be displayed on the graph
        @param show_range: a boolean which indicates if the minimum and maximum
        values should be displayed on the graph
        """

        counts = data_frame.groupby(column).size()
        print(counts)
        mean = data_frame.mean()[column]
        std = data_frame.std()[column]
        median = data_frame.median()[column]

        label = 'mean=' + str(mean) + '\nmedian=' + str(
            median) + '\nstd=' + str(std)

        if show_total:
            total = data_frame.sum()[column]
            label = label + '\ntotal=' + str(total)

        if show_range:
            min_value = data_frame.min()[column]
            max_value = data_frame.max()[column]
            label = label + '\nrange=[' + str(min_value) + ', ' + str(
                max_value) + ']'

        fig, ax = plt.subplots(1)

        counts.plot(kind=plot_type, rot=0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        # these are matplotlib.patch.Patch properties
        properties = dict(boxstyle='round', facecolor='wheat', alpha=0.95)

        ax.text(0.05, 0.95, label, fontsize=14, transform=ax.transAxes,
                verticalalignment='top', bbox=properties)

        return ax

    @staticmethod
    def data_frame_to_csv(data_frame, file_name):
        data_frame.to_csv(file_name, sep='|', encoding='utf8')

    @staticmethod
    def count_series(series):
        counts = {}
        for record in series:
            for item in record:
                if item in counts:
                    counts[item] += record[item]
                else:
                    counts[item] = record[item]
        return counts
