__author__ = 'franpena'

import pylab as pl
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]
diabetes_X_test = diabetes_X_temp[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
pl.scatter(diabetes_X_test, diabetes_y_test, color='black')
pl.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
        linewidth=3)

pl.xticks(())
pl.yticks(())

#pl.show()

print(diabetes_X_test)
print(regr.predict(diabetes_X_test))



regresion = linear_model.LinearRegression()

datosX = [[1001], [1002], [1003], [1004]]
datosY = [4, 8, 12, 26]

regresion.fit(datosX, datosY)
#print regresion.coef_

#pl.scatter(datosX, datosY, color='black')
#pl.plot(datosX, regr.predict(datosY), color='blue',
#        linewidth=3)

#pl.xticks(())
#pl.yticks(())

#pl.show()

'''

from numpy import arange, array, ones, linalg
from pylab import plot, show
from sklearn import datasets, linear_model

clf = linear_model.LinearRegression()
clf.fit([[0, 1], [1, 1], [2, 1]], [4, 8, 12])

print(clf.coef_)

xi = arange(0, 9)
A = array([xi, ones(9)])
# linearly generated sequence
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
w = linalg.lstsq(A.T, y)[0]  # obtaining the parameters

# plotting the line
line = w[0] * xi + w[1]  # regression line
plot(xi, line, 'r-', xi, y, 'o')
#show()

'''
