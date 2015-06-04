import string
import nltk
import re

__author__ = 'fpena'

def tag_words(text):

    # Remove double whitespaces
    paragraph = re.sub("\s\s+", " ", text)

    # Split the paragraph in sentences
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(paragraph)

    tagged_words = []

    for sentence in sentences:
        # words = [word.strip(string.punctuation) for word in sentence.split()]
        words = [word for word in nltk.word_tokenize(sentence)]

        # Lower-case the first word of every sentence so its not confused with a
        # proper noun
        if not words:
            continue

        words[0] = words[0].lower()
        words = filter(None, words)
        tagged_words.extend(nltk.pos_tag(words))

    return tagged_words

def get_nouns(word_tags):
    return [word for (word, tag) in word_tags if tag.startswith('N')]
