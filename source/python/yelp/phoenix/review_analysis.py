from sklearn import linear_model
from etl import ETLUtils
from yelp.phoenix.review_etl import ReviewETL
from sklearn.cross_validation import KFold
import numpy as np

__author__ = 'franpena'


class ReviewAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def simple_lineal_regression(file_path):
        records = ReviewETL.load_file(file_path)
        data = [[record['review_count']] for record in records]
        ratings = [record['stars'] for record in records]

        num_testing_records = int(len(ratings) * 0.8)
        training_data = data[:num_testing_records]
        testing_data = data[num_testing_records:]
        training_ratings = ratings[:num_testing_records]
        testing_ratings = ratings[num_testing_records:]

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(training_data, training_ratings)

        # The coefficients
        print('Coefficients: \n', regr.coef_)
        print('Intercept: \n', regr.intercept_)
        # The root mean square error
        print("RMSE: %.2f"
              % (np.mean(
            (regr.predict(testing_data) - testing_ratings) ** 2)) ** 0.5)

        print(
            'Variance score: %.2f' % regr.score(testing_data, testing_ratings))

        # Plot outputs
        import pylab as pl

        pl.scatter(testing_data, testing_ratings, color='black')
        pl.plot(testing_data, regr.predict(testing_data), color='blue',
                linewidth=3)

        pl.xticks(())
        pl.yticks(())

        pl.show()

    @staticmethod
    def multiple_lineal_regression(file_path):
        records = ReviewETL.load_file(file_path)
        ratings = np.array([record['stars'] for record in records])
        ETLUtils.drop_fields(['stars'], records)
        data = np.array([record.values() for record in records])

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(data, ratings)

        model = linear_model.LinearRegression(fit_intercept=True)
        model.fit(data, ratings)
        p = np.array([model.predict(xi) for xi in data])
        e = p - ratings

        total_error = np.dot(e, e)
        rmse_train = np.sqrt(total_error / len(p))

        kf = KFold(len(data), n_folds=10)
        err = 0
        for train, test in kf:
            model.fit(data[train], ratings[train])
            p = np.array([model.predict(xi) for xi in data[test]])
            e = p - ratings[test]
            err += np.dot(e, e)


        rmse_10cv = np.sqrt(err / len(data))
        print('RMSE on training: {}'.format(rmse_train))
        print('RMSE on 10-fold CV: {}'.format(rmse_10cv))


data_folder = 'E:/UCC/Thesis/datasets/yelp_phoenix_academic_dataset/'
business_file_path = data_folder + 'yelp_academic_dataset_business.json'

ReviewAnalysis.simple_lineal_regression(business_file_path)
ReviewAnalysis.multiple_lineal_regression(business_file_path)
