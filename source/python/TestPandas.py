

import pandas as pd
import re
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor


dataFolder =  'E:/UCC/Thesis/datasets/NYC_Schools/'

# Load the data
dsProgReports = pd.read_csv(dataFolder + 'School_Progress_Reports_-_All_Schools_-_2009-10.csv')
dsDistrict = pd.read_csv(dataFolder + 'School_District_Breakdowns.csv')
dsClassSize = pd.read_csv(dataFolder + '2009-10_Class_Size_-_School-level_Detail.csv')
dsAttendEnroll = pd.read_csv(dataFolder + 'School_Attendance_and_Enrollment_Statistics_by_District__2010-11_.csv')[:-2] #last two rows are bad
dsSATs = pd.read_csv(dataFolder + 'SAT__College_Board__2010_School_Level_Results.csv') # Dependent


#print(pd.DataFrame(data=[dsProgReports['DBN'].take(range(5)), dsSATs['DBN'].take(range(5)), dsClassSize['SCHOOL CODE'].take(range(5))]))


#Strip the first two characters off the DBNs so we can join to School Code
dsProgReports.DBN = dsProgReports.DBN.map(lambda x: x[2:])
dsSATs.DBN = dsSATs.DBN.map(lambda x: x[2:])

#We can now see the keys match
#print(pd.DataFrame(data=[dsProgReports['DBN'].take(range(5)), dsSATs['DBN'].take(range(5)), dsClassSize['SCHOOL CODE'].take(range(5))]))

#Show the key mismatchs
#For variety's sake, using slicing ([:3]) syntax instead of .take()
#print(pd.DataFrame(data=[dsProgReports['DISTRICT'][:3], dsDistrict['JURISDICTION NAME'][:3], dsAttendEnroll['District'][:3]]))


#Extract well-formed district key values
#Note the astype(int) at the end of these lines to coerce the column to a numeric type

dsDistrict['JURISDICTION NAME'] = dsDistrict['JURISDICTION NAME'].map(lambda x: re.match( r'([A-Za-z]*\s)([0-9]*)', x).group(2)).astype(int)
dsAttendEnroll.District = dsAttendEnroll.District.map(lambda x: x[-2:]).astype(int)

#We can now see the keys match
pd.DataFrame(data=[dsProgReports['DISTRICT'][:3], dsDistrict['JURISDICTION NAME'][:3], dsAttendEnroll['District'][:3]])


#Reindexing
dsProgReports = dsProgReports.set_index('DBN')
dsDistrict = dsDistrict.set_index('JURISDICTION NAME')
dsClassSize = dsClassSize.set_index('SCHOOL CODE')
dsAttendEnroll = dsAttendEnroll.set_index('District')
dsSATs = dsSATs.set_index('DBN')

#We can see the bad value
#print(dsSATs['Critical Reading Mean'].take(range(5)))


#Now we filter it out

#We create a boolean vector mask. Open question as to whether this semantically ideal...
mask = dsSATs['Number of Test Takers'].map(lambda x: x != 's')
dsSATs = dsSATs[mask]
#Cast fields to integers. Ideally we should not need to be this explicit.
dsSATs['Number of Test Takers'] = dsSATs['Number of Test Takers'].astype(int)
dsSATs['Critical Reading Mean'] = dsSATs['Critical Reading Mean'].astype(int)
dsSATs['Mathematics Mean'] = dsSATs['Mathematics Mean'].astype(int)
dsSATs['Writing Mean'] = dsSATs['Writing Mean'].astype(int)

#We can see those values are gone
dsSATs['Critical Reading Mean'].take(range(5))


#The shape of the data
#print dsClassSize.columns
#print dsClassSize.take([0,1,10]).values



#Extracting the Pupil-Teacher Ratio

#Take the column
dsPupilTeacher = dsClassSize.filter(['SCHOOLWIDE PUPIL-TEACHER RATIO'])
#And filter out blank rows
mask = dsPupilTeacher['SCHOOLWIDE PUPIL-TEACHER RATIO'].map(lambda x: x > 0)
dsPupilTeacher = dsPupilTeacher[mask]
#Then drop from the original dataset
dsClassSize = dsClassSize.drop('SCHOOLWIDE PUPIL-TEACHER RATIO', axis=1)

#Drop non-numeric fields
dsClassSize = dsClassSize.drop(['BORO','CSD','SCHOOL NAME','GRADE ','PROGRAM TYPE',\
'CORE SUBJECT (MS CORE and 9-12 ONLY)','CORE COURSE (MS CORE and 9-12 ONLY)',\
'SERVICE CATEGORY(K-9* ONLY)','DATA SOURCE'], axis=1)

#Build features from dsClassSize
#In this case, we'll take the max, min, and mean
#Semantically equivalent to select min(*), max(*), mean(*) from dsClassSize group by SCHOOL NAME
#Note that SCHOOL NAME is not referenced explicitly below because it is the index of the dataframe
grouped = dsClassSize.groupby(level=0)
dsClassSize = grouped.aggregate(np.max).\
    join(grouped.aggregate(np.min), lsuffix=".max").\
    join(grouped.aggregate(np.mean), lsuffix=".min", rsuffix=".mean").\
    join(dsPupilTeacher)

#print dsClassSize.columns


mask = dsProgReports['SCHOOL LEVEL*'].map(lambda x: x == 'High School')
dsProgReports = dsProgReports[mask]

final = dsSATs.join(dsClassSize).\
join(dsProgReports).\
merge(dsDistrict, left_on='DISTRICT', right_index=True).\
merge(dsAttendEnroll, left_on='DISTRICT', right_index=True)

final.dtypes[final.dtypes.map(lambda x: x=='object')]

#print(final)

#Just drop string columns.
#In theory we could build features out of some of these, but it is impractical here
final = final.drop(['School Name','SCHOOL','PRINCIPAL','SCHOOL LEVEL*','PROGRESS REPORT TYPE'],axis=1)

#Remove % signs and convert to float
final['YTD % Attendance (Avg)'] = final['YTD % Attendance (Avg)'].map(lambda x: x.replace("%","")).astype(float)

#The last few columns we still have to deal with
final.dtypes[final.dtypes.map(lambda x: x=='object')]


gradeCols = ['2009-2010 OVERALL GRADE','2009-2010 ENVIRONMENT GRADE','2009-2010 PERFORMANCE GRADE','2009-2010 PROGRESS GRADE','2008-09 PROGRESS REPORT GRADE']

grades = np.unique(final[gradeCols].values) #[nan, A, B, C, D, F]

for c in gradeCols:
    for g in grades:
        final = final.join(pd.Series(data=final[c].map(lambda x: 1 if x is g else 0), name=c + "_is_" + str(g)))

final = final.drop(gradeCols, axis=1)

print(final)

#Uncomment to generate csv files
#final.drop(['Critical Reading Mean','Mathematics Mean','Writing Mean'],axis=1).to_csv(dataFolder + 'train.csv')
#final.filter(['Critical Reading Mean','Mathematics Mean','Writing Mean']).to_csv(dataFolder + 'target.csv')

final = final.dropna(axis=0)



target = final.filter(['Critical Reading Mean'])

#We drop all three dependent variables because we don't want them used when trying to make a prediction.

train = final.drop(['Critical Reading Mean','Writing Mean','Mathematics Mean'],axis=1)

#model = RandomForestRegressor(n_estimators=100, n_jobs=-1, compute_importances = True)
model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

#model.fit(train, target)



#predictions = np.array(model.predict(train))

#rmse = math.sqrt(np.mean((np.array(target.values) - predictions)**2))

#imp = sorted(zip(train.columns, model.feature_importances_), key=lambda tup: tup[1], reverse=True)



#print "RMSE: " + str(rmse)

#print "10 Most Important Variables:" + str(imp[:10])

