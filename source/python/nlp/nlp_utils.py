import nltk


def get_sentences(text):
    """
    Returns a list with the sentences there are in the given text

    :type text: str
    :param text: just a text
    :rtype: list[str]
    :return: a list with the sentences there are in the given text
    """
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return sentence_tokenizer.tokenize(text)


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
        words.extend([word for word in nltk.tokenize.word_tokenize(sentence)])
    return words


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
    tokenized_sentences = [get_words(sent.lower()) for sent in sentences]
    if tagger is None:
        tagger = nltk.PerceptronTagger()

    tagged_words = []
    for sent in tokenized_sentences:
        tagged_words.extend(tagger.tag(sent))
    return tagged_words


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


