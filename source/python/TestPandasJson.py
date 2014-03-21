
import numpy as np
from pandas import*
import json

#dataFolder = 'E:/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/'
dataFolder = 'E:/UCC/Thesis/datasets/pydata-book-master/'
path = dataFolder + 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'

#print open(path).readline()

records = [json.loads(line) for line in open(path)]

# Accessing indiviual values: string level parsing
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
time_zones[0:10]




#sequence = time_zones
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts



counts = get_counts(time_zones)
counts['America/New_York']
len(time_zones)
time_zones.sort()


from collections import defaultdict

def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]


top_counts(counts)
value_key_pairs = [(count, ts) for ts, count in counts.items()]
value_key_pairs.sort()
value_key_pairs[-10:]


# Python data analytics library
import pandas as pd
from pandas import DataFrame, Series

# Inserting all records stored in form of lists in to 'pandas DataFrame'
frame = DataFrame(records)


# Series object is returned by frame['tz'] has a method "value_counts" that gives us counts for this particular object
tz_counts = frame['tz'].value_counts()
tz_counts[:10]


# Little Data munging to fill in a substitute value for unknown and missing time zone data in the records.
# The fillna function can replace missing (NA) values and unknown(empty strings) values can be replaced by boolean array indexing
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'

tz_counts = clean_tz.value_counts()
tz_counts[:50]

# PLot a bar graph after data munging, for top 10 values and rotate by 90 deg.
#import pylab as pl
import matplotlib.pyplot as plt
tz_counts[:10].plot(kind='barh', rot=0)
plt.show()


#print tz_counts.plot(kind='barh')





#pandas.read_json(dataFolder + 'yelp_academic_dataset_business.json')
#pandas.read_json(dataFolder + 'yelp_academic_dataset_checkin.json')
#pandas.read_json(dataFolder + 'yelp_academic_dataset_review.json')
#pandas.read_json(dataFolder + 'yelp_academic_dataset_tip.json')
#pandas.read_json(dataFolder + 'yelp_academic_dataset_user.json')

