import numpy
from sklearn.decomposition import nmf
from sklearn.externals import joblib

from utils.constants import Constants


def load_nmf_factors(in_path):
    """
    Load a NMF factorization result using Joblib.
    """
    (W, H, doc_ids, terms) = joblib.load(in_path)
    return W, H, doc_ids, terms


def load_tfidf(in_path):
    """
    Load a pre-processed scikit-learn corpus and associated metadata using
    Joblib.
    """
    tfidf = joblib.load(in_path)
    return tfidf


class NmfTopicExtractor:

    def __init__(self, records):

        self.records = records
        self.num_topics = Constants.TOPIC_MODEL_NUM_TOPICS
        self.tfidf_vectorizer = None
        self.document_topic_matrix = None
        self.topic_term_matrix = None
        self.terms = None

    def load_trained_data(self):

        file_path = Constants.ENSEMBLED_RESULTS_FOLDER + \
            "factors_final_k%02d.pkl" % self.num_topics
        W, H, doc_ids, terms = load_nmf_factors(file_path)
        self.topic_term_matrix = H
        self.document_topic_matrix = W
        self.terms = terms

        topic_model_corpus_folder = \
            Constants.CACHE_FOLDER + 'topic_models/corpus/'
        tfidf_file_path = Constants.generate_file_name(
            'topic_ensemble_corpus', '', topic_model_corpus_folder,
            None, None, False)[:-1] + '_tfidf.pkl'

        self.tfidf_vectorizer = load_tfidf(tfidf_file_path)

        # print('tfidf vectorizer', self.tfidf_vectorizer)

        print "Loaded factor W of size %s and factor H of size %s" % (
            str(self.document_topic_matrix.shape),
            str(self.topic_term_matrix.shape)
        )

    def assign_topic_distribution(self):

        corpora = \
            [' '.join(record[Constants.BOW_FIELD]) for record in self.records]
        document_term_matrix = \
            self.tfidf_vectorizer.transform(corpora)
        document_topic_matrix, _, _ = nmf.non_negative_factorization(
            document_term_matrix, H=self.topic_term_matrix, init='nndsvd',
            n_components=self.num_topics, regularization='both',
            max_iter=Constants.TOPIC_MODEL_ITERATIONS, update_H=False)

        for record_index in range(len(self.records)):
            record = self.records[record_index]
            record[Constants.TOPICS_FIELD] = \
                [(i, document_topic_matrix[record_index][i])
                 for i in range(self.num_topics)]

    def print_topic(self, topic_index, num_terms=10):
        top_indices = numpy.argsort(
            self.topic_term_matrix[topic_index, :])[::-1][0:num_terms]
        term_ranking = [
            '%.3f*%s' % (self.topic_term_matrix[topic_index][i], self.terms[i])
            for i in top_indices
        ]

        topic_string = " + ".join(term_ranking)
        # print("Topic %d: %s" % (topic_index, topic_string))
        return topic_string

    def print_topic_model(self, num_terms=10):

        return [
            self.print_topic(topic_id, num_terms)
            for topic_id in range(self.num_topics)
        ]


def print_topic(topic_index, topic_term_matrix, terms, num_topics, num_terms=10):
    top_indices = numpy.argsort(
        topic_term_matrix[topic_index, :])[::-1][0:num_terms]
    term_ranking = [
        '%.3f*%s' % (topic_term_matrix[topic_index][i], terms[i])
        for i in top_indices
        ]

    topic_string = " + ".join(term_ranking)
    # print("Topic %d: %s" % (topic_index, topic_string))
    return topic_string


def print_topic_model(topic_term_matrix, terms, num_topics, num_terms=10):
    return [
        print_topic(topic_id, topic_term_matrix, terms, num_terms)
        for topic_id in range(num_topics)
        ]
