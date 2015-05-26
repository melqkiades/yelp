import time
import nltk
from topicmodeling.context import context_utils
from topicmodeling.context.review import Review
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora
from gensim.models import ldamodel


__author__ = 'fpena'

class LdaBasedContext:

    def __init__(self, text_reviews):
        self.text_reviews = text_reviews
        self.alpha = 0.005
        self.beta = 1.0
        self.reviews = None
        self.specific_reviews = None
        self.generic_reviews = None
        self.all_nouns = None
        self.all_senses = None
        self.sense_groups = None

    def init_reviews(self):

        print('init_reviews', time.strftime("%H:%M:%S"))

        self.reviews = []
        self.specific_reviews = []
        self.generic_reviews = []

        for text_review in self.text_reviews:
            self.reviews.append(Review(text_review))

        text_specific_reviews, text_generic_reviews =\
            context_utils.cluster_reviews(self.text_reviews)

        for text_review in text_specific_reviews:
            self.specific_reviews.append(Review(text_review))
        for text_review in text_generic_reviews:
            self.generic_reviews.append(Review(text_review))

        self.all_nouns = context_utils.get_all_nouns(self.reviews)

def discover_topics(reviews):

    tokenizer = RegexpTokenizer(r'\w+')
    cachedStopWords = set(stopwords.words("english"))
    body = []
    processed = []

    # remove common words and tokenize
    # texts = [[word for word in document.lower().split() if word not in stopwords.words('english')]
    #          for document in reviews]

    for i in range(0, len(reviews)):
        body.append(reviews[i].lower())

    for entry in body:
        row = tokenizer.tokenize(entry)
        processed.append([word for word in row if word not in cachedStopWords])

    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(2, 0.6)

    # dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in processed]

    # I can print out the documents and which is the most probable topics for each doc.
    lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=50)
    corpus_lda = lda[corpus]
    #
    # for l, t in izip(corpus_lda, corpus):
    #   print l, "#", t
    for l in corpus_lda:
        print(l)
    for topic in lda.show_topics(num_topics=50):
        print(topic)

    return corpus_lda, lda


documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

def main():
    reviews_file = "/Users/fpena/tmp/yelp_academic_dataset_review-mid.json"
    my_reviews = context_utils.load_reviews(reviews_file)
    print("reviews:", len(my_reviews))

    discover_topics(my_reviews)

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)
