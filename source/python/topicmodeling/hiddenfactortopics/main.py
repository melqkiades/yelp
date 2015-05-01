
__author__ = 'fpena'

import sys
sys.path.append('/Users/fpena/UCC/Thesis/projects/yelp/source/python')


from topicmodeling.hiddenfactortopics.corpus import Corpus

my_file = '/Users/fpena/UCC/Thesis/external/McAuley2013/code_RecSys13/Arts.votes'
my_corpus = Corpus(my_file, 0)
my_corpus.process_reviews(my_file, 0)
my_corpus.build_votes(my_file, 0)
