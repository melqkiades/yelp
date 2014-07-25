__author__ = 'fpena'


class MeanAbsoluteError(object):
    def __init__(self):
        self.errors = []

    def add(self, expected, predicted):
        if expected is not None and predicted is not None:
            self.errors.append(abs(expected - predicted))

    def compute(self):
        return MeanAbsoluteError.compute_list(self.errors)

    @staticmethod
    def compute_list(errors):
        """
        Calculates the mean average error for the predicted rating

        :param errors: a list
        :return: the mean average error after predicting all the overall ratings
        """
        num_ratings = 0.
        total_error = 0.

        for error in errors:
            if error is not None:
                total_error += error
                num_ratings += 1

        if num_ratings == 0:
            return None

        mean_absolute_error = total_error / num_ratings
        return mean_absolute_error

