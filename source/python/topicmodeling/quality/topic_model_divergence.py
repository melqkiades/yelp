import numpy
import time
from scipy import stats

from topicmodeling.nmf_topic_extractor import NmfTopicExtractor
from utils.constants import Constants


def calculate_divergence(document_term_matrix, document_topic_matrix, topic_term_matrix):
    corpus_size = document_term_matrix.shape[0]

    # dtm = document_term_matrix.todense()
    print(document_term_matrix.shape)

    # document_lengths = numpy.array([
    #         sum(document_term_matrix[doc_id]) for doc_id in range(corpus_size)])
    document_lengths = numpy.squeeze(numpy.asarray(document_term_matrix.sum(axis=1)))
    # print(document_lengths.shape)
    # print(document_lengths)
    normalized_document_lengths = numpy.linalg.norm(document_lengths)

    c_m1 = numpy.linalg.svd(topic_term_matrix, compute_uv=False)

    c_m2 = document_lengths.dot(document_topic_matrix)
    c_m2 += 0.0001  # we need this to prevent components equal to zero
    c_m2 /= normalized_document_lengths

    # print('c_m1:', c_m1)
    # print('c_m2:', c_m2)

    return my_symmetric_kl(c_m1, c_m2)


def symmetric_kl(distrib_p, distrib_q):
    return numpy.sum([stats.entropy(distrib_p, distrib_q), stats.entropy(distrib_p, distrib_q)])


def KL(a, b):
    return numpy.sum(numpy.where(a != 0, a * numpy.log(a / b), 0))
    # return numpy.sum(a * numpy.log(a / b))


def my_kl(a, b):

    k = len(a)

    total = 0.0
    for i in range(k):
        total += a[i] * numpy.log(a[i] / b[i])

    return total


def my_symmetric_kl(a, b):

    return my_kl(a, b) + my_kl(b, a)


def test():
    document_term_matrix = NmfTopicExtractor.load_document_term_matrix()

    results = []

    my_list = [2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 20, 22, 23, 25, 28, 30]

    for i in my_list:
        Constants.update_properties({Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: i})
        topic_model = NmfTopicExtractor()
        topic_model.load_trained_data()
        # topic_model_string = topic_model.print_topic_model(5)

        # i = 1
        # for topic in topic_model_string:
        #     print('Topic: %d:\t%s' % (i, topic))
        #     i += 1


        document_topic_matrix = topic_model.document_topic_matrix
        topic_term_matrix = topic_model.topic_term_matrix

        divergence = calculate_divergence(document_term_matrix, document_topic_matrix, topic_term_matrix)

        results.append((Constants.TOPIC_MODEL_NUM_TOPICS, divergence))

        print('Num topics: %d, Divergence: %f' % (Constants.TOPIC_MODEL_NUM_TOPICS, divergence))

    for num_topic, divergence in results:
        print('%d %f' % (num_topic, divergence))


def main():
    test()


# values1 = numpy.asarray([1.346112, 1.337432, 1.246655], dtype=numpy.float)
# values2 = numpy.asarray([1.033836, 1.082015, 1.117323], dtype=numpy.float)
#
# # Output: 0.775279624079
#
#
# print(KL(values1, values2))
# print(KL(values2, values1))
# print(KL(values1, values2) + KL(values2, values1))
# print(symmetric_kl(values1, values2))
# print(my_kl(values1, values2))
# print(my_kl(values2, values1))
# print(my_symmetric_kl(values1, values2))


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)

