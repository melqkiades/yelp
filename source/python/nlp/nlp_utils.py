import gensim
import nltk
from gensim.utils import lemmatize
from pattern.text.en import parse


def get_sentences(text):
    """
    Returns a list with the sentences there are in the given text

    :type text: str
    :param text: just a text
    :rtype: list[str]
    :return: a list with the sentences there are in the given text
    """
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    newline_re = nltk.re.compile('\n')
    paragraphs = newline_re.split(text)
    sentences = []
    for paragraph in paragraphs:
        # sentences.append(paragraph)
        sentences.extend(sentence_tokenizer.tokenize(paragraph))

    return sentences
    # return sentence_tokenizer.tokenize(text)



def get_words(text):
    """
    Splits the given text into words and returns them

    :type text: str
    :param text: just a text. It must be in english.
    :rtype: list[str]
    :return: a list with the words there are in the given text
    """
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)

    words = []

    for sentence in sentences:
        words.extend(get_words_from_sentence(sentence))

    return words


def get_words_from_sentence(sentence):
    """
    Splits the given sentence into words and returns them

    :type sentence: str
    :param sentence: just a text. It must be in english and it must be only one
    sentence, otherwise it may bring down the quality of the results.
    :rtype: list[str]
    :return: a list with the words there are in the given sentence
    """
    return nltk.tokenize.word_tokenize(sentence)


def tag_words(text, tagger=None):
    """
    Tags the words contained in the given text using part-of-speech tags. The
    text is split into sentences and it returns a list of lists with the tagged
    words. One list for every sentence.

    :param tagger: a part-of-speech tagger. This parameter is useful in order to
    avoid the initialization of the tagger every time this method is called,
    since the initialization can take a long time.
    :param text: the text to tag
    :return: a list of lists with pairs, in the form of (word, tag)
    """
    sentences = get_sentences(text)
    tokenized_sentences = [
        get_words_from_sentence(sent.lower()) for sent in sentences]
    if tagger is None:
        tagger = nltk.PerceptronTagger()

    tagged_words = []
    for sent in tokenized_sentences:
        tagged_words.extend(tagger.tag(sent))
    return tagged_words


def lemmatize_text(text):
    """
    Tags the words contained in the given text using part-of-speech tags. The
    text is split into sentences and it returns a list of lists with the tagged
    words. One list for every sentence.

    :param tagger: a part-of-speech tagger. This parameter is useful in order to
    avoid the initialization of the tagger every time this method is called,
    since the initialization can take a long time.
    :param text: the text to tag
    :return: a list of lists with pairs, in the form of (word, tag)
    """
    sentences = get_sentences(text)
    # tokenized_sentences = [
    #     get_words_from_sentence(sent.lower()) for sent in sentences]

    lemmatized_words = []
    for sentence in sentences:
        lemmatized_words.extend(lemmatize_sentence(
            sentence, nltk.re.compile(''),
            min_length=1, max_length=100))
    return lemmatized_words


def lemmatize_sentence(content, allowed_tags=nltk.re.compile(''),
               stopwords=frozenset(), min_length=1, max_length=100):


    """
    This function is only available when the optional 'pattern' package is installed.

    Use the English lemmatizer from `pattern` to extract UTF8-encoded tokens in
    their base form=lemma, e.g. "are, is, being" -> "be" etc.
    This is a smarter version of stemming, taking word context into account.

    Only considers nouns, verbs, adjectives and adverbs by default (=all other lemmas are discarded).

    >>> lemmatize('Hello World! How is it going?! Nonexistentword, 21')
    ['world/NN', 'be/VB', 'go/VB', 'nonexistentword/NN']

    >>> lemmatize('The study ranks high.')
    ['study/NN', 'rank/VB', 'high/JJ']

    >>> lemmatize('The ranks study hard.')
    ['rank/NN', 'study/VB', 'hard/RB']

    """

    # tokenization in `pattern` is weird; it gets thrown off by non-letters,
    # producing '==relate/VBN' or '**/NN'... try to preprocess the text a little
    # FIXME this throws away all fancy parsing cues, including sentence structure,
    # abbreviations etc.
    content = gensim.utils.u(' ').join(
        gensim.utils.tokenize(content, lower=True, errors='ignore'))

    parsed = parse(content, lemmata=True, collapse=False)
    result = []
    for sentence in parsed:
        for token, tag, _, _, lemma in sentence:
            if min_length <= len(lemma) <= max_length and not lemma.startswith(
                    '_') and lemma not in stopwords:
                if allowed_tags.match(tag):
                    result.append((token, tag, lemma))
    return result


def count_verbs(tags_count):
    """
    Receives a dictionary with part-of-speech tags as keys and counts as values,
    returns the total number of verbs that appear in the dictionary

    :type tags_count: dict
    :param tags_count: a dictionary with part-of-speech tags as keys and counts
    as values
    :rtype : int
    :return: the total number of verbs that appear in the dictionary
    """
    total_verbs = 0
    for key in tags_count.keys():
        if key.startswith('VB'):
            total_verbs += tags_count[key]

    return total_verbs


