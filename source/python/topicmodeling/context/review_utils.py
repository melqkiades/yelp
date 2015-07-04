import nltk
import re

__author__ = 'fpena'


def tag_words(text):
    """
    Tags the words contained in the given text using part-of-speech tags

    :param text: the text to tag
    :return: a list with pairs, in the form of (word, tag)
    """
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


def get_nouns(pos_tag_list):
    """
    Receives a list with tagged words (using part-of-speech), filters it
    returning only the words that are nouns

    :type pos_tag_list: list[(str, str)]
    :param pos_tag_list: a list with part-of-speech tags
    :rtype: list[(str, str)]
    :return: a list with only the words that are nouns
    """
    return [word for (word, tag) in pos_tag_list if tag.startswith('N')]
