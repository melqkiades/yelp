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
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
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
            simplified = [(word, simplify_wsj_tag(tag)) for word, tag in tagged_tip]
            for tagged_word in simplified:
                tags.add(tagged_word[1])
            index += 1
            print(index)

        print(tags)

    @staticmethod
    def load_sent_word_net():

        sent_scores = collections.defaultdict(list)

        with open("../../../../../../datasets/SentiWordNet_3.0.0_20130122.txt", "r") as csvfile:
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

        tagged_words = TipPosTagger.tag_text(sentence)

        for noun_index, (noun, noun_tag) in enumerate(tagged_words):
            if noun_tag.startswith("NN"):
                self.grade_noun(noun, noun_index, tagged_words)

    def grade_noun(self, noun, noun_index, tagged_words):
        for adjective_index, (adjective, adjective_tag) in enumerate(tagged_words):
            if adjective_tag.startswith("JJ"):
                weight = TipPosTagger.calculate_weight(noun_index, adjective_index)
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
        if index1 == index2:
            return 0
        return 1.0 / math.fabs(index1 - index2)



data_folder = '../../../../../../datasets/yelp_phoenix_academic_dataset/'
tip_file_path = data_folder + 'yelp_academic_dataset_tip.json'
my_records = ETLUtils.load_json_file(tip_file_path)
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['hW0Ne_HTHEAgGF1rAdmR-g'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['JokKtdXU7zXHcr20Lrk29A'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['0UZ31UTcOLRKuqPqPe-VBA'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['aRkYtXfmEKYG-eTDf_qUsw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['-sC66z4SO3tR7nFCjfQwuQ'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['EWMwV5V9BxNs_U6nNVMeqw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['L9UYbtAUOcfTgZFimehlXw'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['uFJwKlHL6HyHSJmORO8-5w'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['WS1z1OAR0tRl4FsjdTGUFQ'])
business_records = ETLUtils.filter_records(my_records, 'business_id', ['FURgKkRFtMK5yKbjYZVVwA'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['Gq092IH6eZqhAXwtXcwc6A'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['R8VwdLyvsp9iybNqRvm94g'])
# business_records = ETLUtils.filter_records(my_records, 'business_id', ['uKSX1n1RoAzGq4bV8GPHVg'])
my_tips = [my_record['text'] for my_record in business_records]
# TipPosTagger.process_tips(my_tips[:1000])


my_text = "The burgers are very good. The service is bad."
my_tags = TipPosTagger.tag_text(my_text)
simp = [(my_word, simplify_wsj_tag(my_tag)) for my_word, my_tag in my_tags]
print(simp)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
my_sentences = tokenizer.tokenize(my_text)



# print sent_words

tip_pos_tagger = TipPosTagger()
tip_pos_tagger.grade_sentences(my_sentences)
tip_pos_tagger.analyze_tips(my_tips)

sorted_x = sorted(tip_pos_tagger.noun_dictionary.iteritems(), key=operator.itemgetter(1))

print(sorted_x)

# print tip_pos_tagger.calculate_word_score(('worst', 'ADJ'))

# nltk.download()
