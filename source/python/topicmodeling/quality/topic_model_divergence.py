import numpy
import time
from scipy import stats

from etl import ETLUtils
from etl.reviews_preprocessor import ReviewsPreprocessor
from topicmodeling.nmf_topic_extractor import NmfTopicExtractor
from utils.constants import Constants


def calculate_divergence(
        document_term_matrix, document_topic_matrix, topic_term_matrix):

    print(document_term_matrix.shape)

    document_lengths =\
        numpy.squeeze(numpy.asarray(document_term_matrix.sum(axis=1)))
    normalized_document_lengths = numpy.linalg.norm(document_lengths)

    c_m1 = numpy.linalg.svd(topic_term_matrix, compute_uv=False)

    c_m2 = document_lengths.dot(document_topic_matrix)
    c_m2 += 0.0001  # we need this to prevent components equal to zero
    c_m2 /= normalized_document_lengths

    return my_symmetric_kl(c_m1, c_m2)


def symmetric_kl(distrib_p, distrib_q):
    return numpy.sum([
        stats.entropy(distrib_p, distrib_q),
        stats.entropy(distrib_p, distrib_q)
    ])


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


def create_topic_models():
    my_list = range(2, 61)

    for i in my_list:
        Constants.update_properties({Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: i})
        reviews_preprocessor = ReviewsPreprocessor(use_cache=True)
        reviews_preprocessor.full_cycle()


def test():
    document_term_matrix = NmfTopicExtractor.load_document_term_matrix()

    results = []

    # my_list = range(2, 31)
    my_list = range(2, 61)

    for i in my_list:
        Constants.update_properties({Constants.TOPIC_MODEL_NUM_TOPICS_FIELD: i})
        topic_model = NmfTopicExtractor()
        topic_model.load_trained_data()

        document_topic_matrix = topic_model.document_topic_matrix
        topic_term_matrix = topic_model.topic_term_matrix

        divergence = calculate_divergence(
            document_term_matrix, document_topic_matrix, topic_term_matrix)

        result = {
            'num_topics': Constants.TOPIC_MODEL_NUM_TOPICS,
            'divergence': divergence,
            Constants.TOPIC_MODEL_TYPE_FIELD: 'ensemble',
            Constants.BUSINESS_TYPE_FIELD: Constants.ITEM_TYPE
        }

        results.append(result)

        print('Num topics: %d, Divergence: %f' %
              (Constants.TOPIC_MODEL_NUM_TOPICS, divergence))

    for result in results:
        print('%d %f' % (result['num_topics'], result['divergence']))

    prefix = Constants.RESULTS_FOLDER + Constants.ITEM_TYPE +\
        '_topic_model_divergence'
    csv_file_path = prefix + '.csv'
    json_file_path = prefix + '.json'
    headers = sorted(results[0].keys())
    ETLUtils.save_csv_file(csv_file_path, results, headers)
    ETLUtils.save_json_file(json_file_path, results)


def main():
    # create_topic_models()
    test()


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
