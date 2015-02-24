import math
import nltk
import operator
from etl import ETLUtils
from nltk.tag.simplify import simplify_wsj_tag
import collections
import csv
import numpy

__author__ = 'franpena'


class TipPosTagger:
    def __init__(self):
        self.noun_dictionary = {}
        self.sentiment_words = TipPosTagger.load_sent_word_net()
        self.sentence_tokenizer = nltk.data.load(
            'tokenizers/punkt/english.pickle')
        self.stemmer = nltk.stem.SnowballStemmer('english')

    @staticmethod
    def tag_text(text):
        words = nltk.word_tokenize(text)
        tagged_words = nltk.pos_tag(words)
        return tagged_words

    @staticmethod
    def process_tips(tips):

        tags = set()
        index = 0
        for tip in tips:
            tagged_tip = TipPosTagger.tag_text(tip)
            simplified = [(word, simplify_wsj_tag(tag)) for word, tag in
                          tagged_tip]
            for tagged_word in simplified:
                tags.add(tagged_word[1])
            index += 1
            print(index)

        print(tags)

    @staticmethod
    def load_sent_word_net():

        sent_scores = collections.defaultdict(list)

        with open("../../../../../../datasets/SentiWordNet_3.0.0_20130122.txt",
                  "r") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for line in reader:
                if line[0].startswith("#"):
                    continue
                if len(line) == 1:
                    continue

                POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
                if len(POS) == 0 or len(ID) == 0:
                    continue
                # print POS,PosScore,NegScore,SynsetTerms
                for term in SynsetTerms.split(" "):
                    # drop #number at the end of every term
                    term = term.split("#")[0]
                    term = term.replace("-", " ").replace("_", " ")
                    key = "%s/%s" % (POS, term.split("#")[0])
                    sent_scores[key].append((float(PosScore), float(NegScore)))
        for key, value in sent_scores.iteritems():
            sent_scores[key] = numpy.mean(value, axis=0)

        return sent_scores

    def calculate_word_score(self, tagged_word):

        """
        Takes a tagged word (in the form of a pair (word, tag)) and returns the
        score associated to that word. In other words it returns how positive
        or negative the word is in the range [-1, 1] where -1 is the most
        negative, 1 is the most positive and 0 is neutral. The tag refers to the
        part of speech tag associated to the word, i.e. noun, verb,
        adjective, etc.

        :rtype : float
        :param tagged_word: the word from which we want the score and its part
        of speech tag
        :return: a float number that represents how positive or negative is the
        word
        """
        word = tagged_word[0].lower()
        tag = tagged_word[1]

        if tag.startswith('JJ'):
            tag = 'a'
        else:
            return 0

        dict_word = tag + '/' + word
        total_score = 0

        if dict_word in self.sentiment_words:
            score = self.sentiment_words[dict_word]
            total_score = score[0] - score[1]

        return total_score

    def grade_sentence(self, sentence):

        """
        Scores all the nouns in the given sentence and stores/updates the scores
        in the noun_dictionary of this class

        :rtype : void
        :param sentence: a string with the sentence to be scored
        """
        tagged_words = TipPosTagger.tag_text(sentence)

        for noun_index, (noun, noun_tag) in enumerate(tagged_words):
            if noun_tag.startswith("NN"):
                self.grade_noun(noun, noun_index, tagged_words)

    def grade_noun(self, noun, noun_index, tagged_words):
        """
        Grades a noun that belongs to the sentence in tagged_words. The noun
        will be graded according to the positivity or negativity of the words
        that surround it

        :rtype : void
        :param noun: the word to be graded
        :param noun_index: the position of the word in the tagged_words array
        (sentence)
        :param tagged_words: an array of pairs in which each pair is in the form
        (word, tag). The concatenation of all the words forms the sentence to
        be analyzed
        """
        for adjective_index, (adjective, adjective_tag) in enumerate(
                tagged_words):
            if adjective_tag.startswith("JJ"):
                weight = TipPosTagger.calculate_weight(noun_index,
                                                       adjective_index)
                score = self.calculate_word_score((adjective, adjective_tag))
                noun = self.stemmer.stem(noun)
                if noun in self.noun_dictionary:
                    self.noun_dictionary[noun] += score * weight
                else:
                    self.noun_dictionary[noun] = score * weight

    def grade_sentences(self, sentences):
        for sentence in sentences:
            self.grade_sentence(sentence)

    def analyze_tips(self, tips):
        for tip in tips:
            sentences = self.sentence_tokenizer.tokenize(tip)
            self.grade_sentences(sentences)

    @staticmethod
    def calculate_weight(index1, index2):
        """
        Calculates the impact that a word which is in position index2 will have
        over the word that is in position index1. For instance, for the
        sentences 'good soup' and 'the soup is good' the word 'good' will have
        more impact over the word 'soup' in the first sentence because its
        closer to it than in the second sentence.

        :rtype : float
        :param index1: the position of the first word (impactee word) in the
        sentence
        :param index2: the position of the second word (impact word) in the
        sentence
        :return: a float with the amount of impact that the second word has over
        the first one. This float is in the range [0, 1] where 0 means no impact
        and 1 means maximum impact
        """
        if index1 == index2:
            return 0
        return 1.0 / math.fabs(index1 - index2)


data_folder = '../../../../../../datasets/yelp_phoenix_academic_dataset/'
tip_file_path = data_folder + 'yelp_academic_dataset_tip.json'
review_file_path = data_folder + 'yelp_academic_dataset_review.json'
my_records = ETLUtils.load_json_file(review_file_path)
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['hW0Ne_HTHEAgGF1rAdmR-g'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['JokKtdXU7zXHcr20Lrk29A'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['0UZ31UTcOLRKuqPqPe-VBA'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['aRkYtXfmEKYG-eTDf_qUsw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['-sC66z4SO3tR7nFCjfQwuQ'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['EWMwV5V9BxNs_U6nNVMeqw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['L9UYbtAUOcfTgZFimehlXw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['uFJwKlHL6HyHSJmORO8-5w'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['WS1z1OAR0tRl4FsjdTGUFQ'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['FURgKkRFtMK5yKbjYZVVwA'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['Gq092IH6eZqhAXwtXcwc6A'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['R8VwdLyvsp9iybNqRvm94g'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['uKSX1n1RoAzGq4bV8GPHVg'])

# business_records = ETLUtils.filter_records(my_records, 'business_id', ['hW0Ne_HTHEAgGF1rAdmR-g'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['VVeogjZya58oiTxK7qUjAQ'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['JokKtdXU7zXHcr20Lrk29A'])
business_records = ETLUtils.filter_records(my_records, 'business_id', ['EWMwV5V9BxNs_U6nNVMeqw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['V1nEpIRmEa1768oj_tuxeQ'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['SDwYQ6eSu1htn8vHWv128g'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['WNy1uzcmm_UHmTyR--o5IA'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['ntN85eu27C04nwyPa8IHtw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['-sC66z4SO3tR7nFCjfQwuQ'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['QnAzW6KMSciUcuJ20oI3Bw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['uKSX1n1RoAzGq4bV8GPHVg'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['YKOvlBNkF4KpUP9q7x862w'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['aRkYtXfmEKYG-eTDf_qUsw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['pwpl-rxwNRQdgqFz_-qMPg'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['3oZcTGb_oDHGwZFiP-7kxQ'])

my_tips = [my_record['text'] for my_record in business_records]
# TipPosTagger.process_tips(my_tips[:1000])


my_text = "The burgers are very good. The service is bad." + \
          "It is a great place to go with friends. I went there with my wife."
my_tags = TipPosTagger.tag_text(my_text)
simp = [(my_word, simplify_wsj_tag(my_tag)) for my_word, my_tag in my_tags]
print(simp)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
my_sentences = tokenizer.tokenize(my_text)



# print sent_words

tip_pos_tagger = TipPosTagger()
tip_pos_tagger.analyze_tips(my_tips)

sorted_x = sorted(tip_pos_tagger.noun_dictionary.iteritems(), key=operator.itemgetter(1))

print(sorted_x[:10])
print(sorted_x[-10:])

for tip in ETLUtils.search_sentences(my_tips, 'pomegranate margarita'): print(tip)

pattern = """
    NP:{<DT>?<JJ.*>*<NN.*>+}
    CNTXT:{<IN>?<DT>?<NN.*>+}
    ADJ:{<JJ|JJR|JJS>?}
"""
c = nltk.RegexpParser(pattern)
my_tokens = nltk.word_tokenize("If its your first time, get the fez burger and you won't be disappointed.")
my_tagged_words = nltk.pos_tag(my_tokens)
t = c.parse(my_tagged_words)
print my_tagged_words

# print tip_pos_tagger.calculate_word_score(('worst', 'ADJ'))
