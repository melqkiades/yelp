from gensim import corpora
from gensim.models import ldamodel
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

__author__ = 'fpena'


def update_reviews_with_topics(topics, reviews):

    if len(topics) != len(reviews):
        raise ValueError("The topics and reviews lists must have the same length")

    index = 0
    for topic in topics:
        reviews[index].topics = topic
        index += 1


def calculate_topic_weighted_frequency(topic, reviews):
    """

    :type topic: int
    :param topic:
    :type reviews: list[Review]
    :param reviews:
    :return:
    """
    num_reviews = 0.0

    for review in reviews:
        for review_topic in review.topics:
            if topic == review_topic[0]:
                num_reviews += 1

    return num_reviews / len(reviews)


def discover_topics(text_reviews, num_topics):

    processed = create_bag_of_words(text_reviews)

    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(2, 0.6)

    # dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in processed]

    # I can print out the documents and which is the most probable topics for each doc.
    lda_model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
    corpus_lda = lda_model[corpus]
    #
    # for l, t in izip(corpus_lda, corpus):
    #   print l, "#", t
    # for l in corpus_lda:
    #     print(l)
    for topic in lda_model.show_topics(num_topics=num_topics, num_words=50):
        print(topic)

    return lda_model


def create_bag_of_words(document_list):
    """
    Creates a bag of words representation of the document list given. It removes
    the punctuation and the stop words.

    :type document_list: list[str]
    :param document_list:
    :rtype: list[list[str]]
    :return:
    """
    tokenizer = RegexpTokenizer(r'\w+')
    cached_stop_words = set(stopwords.words("english"))
    body = []
    processed = []

    # remove common words and tokenize
    # texts = [[word for word in document.lower().split() if word not in stopwords.words('english')]
    #          for document in reviews]

    for i in range(0, len(document_list)):
        body.append(document_list[i].lower())

    for entry in body:
        row = tokenizer.tokenize(entry)
        processed.append([word for word in row if word not in cached_stop_words])

    return processed


my_documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"
]

# print(create_bag_of_words(my_documents))

my_bag_of_words = create_bag_of_words(my_documents)
my_topic_model = discover_topics(my_documents, 3)
my_dictionary = corpora.Dictionary(my_bag_of_words)
my_dictionary.filter_extremes(2, 0.6)
my_corpus = [my_dictionary.doc2bow(text) for text in my_bag_of_words]
# print(my_topic_model[my_corpus])
# for t in my_topic_model[my_corpus]:
#     print(t)
