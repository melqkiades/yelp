import math
from pandas import DataFrame
import time
from DataPlotter import DataPlotter
from tripadvisor.fourcity import extractor
import matplotlib.pyplot as plt

__author__ = 'fpena'


def plot_overall_rating():

    reviews = extractor.pre_process_reviews()
    data_frame = DataFrame(reviews)

    print(data_frame)

    DataPlotter.plot_data(data_frame, 'overall_rating', plot_type='bar', title='Overall Rating')
    DataPlotter.plot_data(data_frame, 'cleanliness_rating', plot_type='bar', title='Cleanliness Rating')
    DataPlotter.plot_data(data_frame, 'location_rating', plot_type='bar', title='Location Rating')
    DataPlotter.plot_data(data_frame, 'rooms_rating', plot_type='bar', title='Rooms Rating')
    DataPlotter.plot_data(data_frame, 'service_rating', plot_type='bar', title='Service Rating')
    DataPlotter.plot_data(data_frame, 'value_rating', plot_type='bar', title='Value Rating')
    plt.show()




def main():
    plot_overall_rating()


start_time = time.time()
main()
end_time = time.time() - start_time
print("--- %s seconds ---" % end_time)