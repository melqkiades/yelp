import json
import itertools

import numpy as np
from pandas import DataFrame
from DataPlotter import DataPlotter
from YelpDataMain import YelpDataMain
import matplotlib.pyplot as plt


dataFolder = 'E:/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/'
business_file_path = dataFolder + 'yelp_academic_dataset_business.json'
checkin_file_path = dataFolder + 'yelp_academic_dataset_checkin.json'
review_file_path = dataFolder + 'yelp_academic_dataset_review.json'
tip_file_path = dataFolder + 'yelp_academic_dataset_tip.json'
user_file_path = dataFolder + 'yelp_academic_dataset_user.json'

records = [json.loads(line) for line in open(checkin_file_path)]

# Inserting all records stored in form of lists in to 'pandas DataFrame'
dataFrame = DataFrame(records)
#print(dataFrame.columns.tolist())
#categories = dataFrame['categories']
#print(categories)
#print(dataFrame['checkin_info'])

checkinDataFrame = DataFrame(dataFrame['checkin_info'])
print(records[0]['checkin_info'])
#print(checkinDataFrame['checkin_info'])
DataPlotter.count_series(checkinDataFrame['checkin_info'])

#merged = list(itertools.chain.from_iterable(categories))
#df = DataFrame(merged, columns=['category'])
#rating_counts = df.groupby('category').size()

#export_data_frame = dataFrame.drop(
#    ['attributes', 'categories', 'full_address', 'hours', 'latitude', 'longitude', 'neighborhoods', 'state'], axis=1)
#DataPlotter.data_frame_to_csv(export_data_frame, dataFolder + 'train.csv')
#export_data_frame = dataFrame.drop(['text'], axis=1)
#DataPlotter.data_frame_to_csv(export_data_frame, dataFolder + 'review.csv')

#YelpDataMain.plot_business_stars()
YelpDataMain.plot_business_reviews()
#YelpDataMain.plot_user_stars()
YelpDataMain.plot_user_reviews()

#checkin_count = DataPlotter.count_series(checkinDataFrame['checkin_info'])
#print checkin_count.items()
#ckDataFrame = DataFrame(checkin_count.items(), None, ['myCol1', 'myCol2'])
#DataPlotter.data_frame_to_csv(ckDataFrame, dataFolder + 'checkins.csv')

# Too long
#YelpDataMain.plot_review_stars()
