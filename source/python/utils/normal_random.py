import math

__author__ = 'fpena'

import numpy as np

class NormalRandom:

    spare = 0
    is_spare_ready = False

    @staticmethod
    def generate():

        # print('is_spare_ready', NormalRandom.is_spare_ready)
        # return np.random.random()

        if NormalRandom.is_spare_ready:
            # print('Inside if')
            NormalRandom.is_spare_ready = False
            return NormalRandom.spare
        else:
            # print('Inside else')
            u = np.random.random() * 2 - 1
            v = np.random.random() * 2 - 1
            s = u * u + v * v

            # print('u', u)
            # print('v', v)

            while s >= 1 or s == 0:
                # print('Inside while')
                u = np.random.random() * 2 - 1
                v = np.random.random() * 2 - 1
                s = u * u + v * v

                # print('u', u)
                # print('v', v)

            mul = math.sqrt(-2.0 * math.log(s) / s)
            NormalRandom.spare = v * mul
            NormalRandom.is_spare_ready = True

            return u * mul


    @staticmethod
    def generate_list(size):

        array = []
        for i in range(size):
            array.append(NormalRandom.generate())

        return np.array(array)


    @staticmethod
    def generate_matrix(rows, columns):

        matrix = []
        for i in range(rows):
            matrix.append(NormalRandom.generate_list(columns))

        # return np.array(matrix, dtype='float32')
        # return np.array(matrix, dtype=object)
        return np.array(matrix)

# np.random.seed(4)
#
# print np.random.rand()
# print NormalRandom.generate_list(10)
# my_matrix = NormalRandom.generate_matrix(3, 3)
# print(my_matrix)
# np.random.seed(4)
# temp_feature = np.random.rand(3, 3)
#
# print(temp_feature)
#
# print(np.array_equal(my_matrix, temp_feature))
# print(np.array_equal(temp_feature, my_matrix))


