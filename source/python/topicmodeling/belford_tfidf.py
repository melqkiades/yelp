#!/usr/bin/env python
"""
Tool to parse a collection of documents, where each file is stored in a separate plain text file.
Sample usage:
python parse-directory.py data/sample-text/ -o sample --tfidf --norm
"""
import os, os.path, sys, codecs, re, unicodedata
import logging as log
from optparse import OptionParser
import cPickle as pickle

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(
        docs, stopwords, min_df=3, min_term_length=2,
        ngram_range=(1, 1), apply_tfidf=True, apply_norm=True,
        lemmatize=False):
    """
    Preprocess a list containing text documents stored as strings.
    """
    token_pattern = re.compile(r"\b\w\w+\b", re.U)

    if lemmatize:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()

    def normalize(x):
        x = x.lower()
        if lemmatize:
            return wnl.lemmatize(x)
        return x

    def custom_tokenizer(s):
        return [normalize(x) for x in token_pattern.findall(s) if (len(x) >= min_term_length and x[0].isalpha() ) ]

    # Build the Vector Space Model, apply TF-IDF and normalize lines to unit
    # length all in one call
    if apply_norm:
        norm_function = "l2"
    else:
        norm_function = None
    tfidf = TfidfVectorizer(
        stop_words=stopwords, lowercase=True, strip_accents="unicode",
        tokenizer=None, use_idf=apply_tfidf, norm=norm_function,
        min_df=min_df, ngram_range=ngram_range)
    X = tfidf.fit_transform(docs)
    terms = []
    # store the vocabulary map
    v = tfidf.vocabulary_
    for i in range(len(v)):
        terms.append("")
    for term in v.keys():
        terms[v[term]] = term
    return (X, terms, tfidf)


def load_stopwords( inpath = "text/stopwords.txt"):
    """
    Load stopwords from a file into a set.
    """
    stopwords = set()
    with open(inpath) as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().lower()
            if len(l) > 0:
                stopwords.add(l)
    return stopwords


def save_tfidf(out_prefix, tfidf):
    """
    Save a pre-processed scikit-learn corpus and associated metadata using
    Joblib.
    """
    tfidf_outpath = "%s_tfidf.pkl" % out_prefix
    joblib.dump(tfidf, tfidf_outpath)

    log.info('TF-IDF path: %s' % tfidf_outpath)

    # pickle.dump(tfidf, open(tfidf_outpath, "wb"))

    # with open(tfidf_outpath, 'wb') as write_file:
    #     pickle.dump(tfidf, write_file, pickle.HIGHEST_PROTOCOL)


def save_corpus(out_prefix, X, terms, doc_ids, classes=None):
    """
    Save a pre-processed scikit-learn corpus and associated metadata using
    Joblib.
    """
    matrix_outpath = "%s.pkl" % out_prefix
    joblib.dump((X, terms, doc_ids, classes), matrix_outpath)


def load_tfidf(in_path):
    """
    Load a pre-processed scikit-learn corpus and associated metadata using
    Joblib.
    """
    # tfidf = joblib.load(in_path)
    with open(in_path, 'rb') as read_file:
        tfidf = pickle.load(read_file)
    return tfidf


def load_corpus(in_path):
    """
    Load a pre-processed scikit-learn corpus and associated metadata using
    Joblib.
    """
    (X, terms, doc_ids, classes) = joblib.load(in_path)
    return (X, terms, doc_ids, classes)


def find_documents(root_path):
    """
    Find all files in the specified directory and its subdirectories, and store
    them as strings in a list.
    """
    filepaths = []
    for dir_path, subFolders, files in os.walk(root_path):
        for filename in files:
            if filename.startswith(".") or filename.startswith("_"):
                continue
            filepath = os.path.join(dir_path, filename)
            filepaths.append(filepath)
    filepaths.sort()
    return filepaths


def read_text(in_path):
    """
    Read and normalize body text from the specified document file.
    """
    http_re = re.compile('https?[:;]?/?/?\S*')
    # read the file
    f = codecs.open(in_path, 'r', encoding="utf8", errors='ignore')
    body = ""
    while True:
        line = f.readline()
        if not line:
            break
        # Remove URIs at this point (Note: this simple regex captures MOST URIs
        # but may occasionally let others slip through)
        normalized_line = re.sub(http_re, '', line.strip())
        if len(normalized_line) > 1:
            body += normalized_line
            body += "\n"
    f.close()
    return body


# --------------------------------------------------------------

def main():
    parser = OptionParser(usage="usage: %prog [options] dir1 dir2 ...")
    parser.add_option("-o", action="store", type="string", dest="prefix",
                      help="output prefix for corpus files", default=None)
    parser.add_option("--df", action="store", type="int", dest="min_df",
                      help="minimum number of documents for a term to appear",
                      default=20)
    parser.add_option("--tfidf", action="store_true", dest="apply_tfidf",
                      help="apply TF-IDF term weight to the document-term matrix")
    parser.add_option("--norm", action="store_true", dest="apply_norm",
                      help="apply unit length normalization to the document-term matrix")
    parser.add_option("--minlen", action="store", type="int",
                      dest="min_doc_length",
                      help="minimum document length (in characters)",
                      default=50)
    parser.add_option("-s", action="store", type="string", dest="stoplist_file",
                      help="custom stopword file path", default=None)
    parser.add_option('-d', '--debug', type="int",
                      help="Level of log output; 0 is less, 5 is all",
                      default=3)
    (options, args) = parser.parse_args()
    if (len(args) < 1):
        parser.error("Must specify at least one directory")
    log.basicConfig(level=max(50 - (options.debug * 10), 10),
                    format='%(asctime)-18s %(levelname)-10s %(message)s',
                    datefmt='%d/%m/%Y %H:%M', )

    # Find all relevant files in directories specified by user
    filepaths = []
    args.sort()
    for in_path in args:
        if os.path.isdir(in_path):
            log.info("Searching %s for documents ..." % in_path)
            for fpath in find_documents(in_path):
                filepaths.append(fpath)
        else:
            if in_path.startswith(".") or in_path.startswith("_"):
                continue
            filepaths.append(in_path)
    log.info("Found %d documents to parse" % len(filepaths))

    # Read the documents
    log.info("Reading documents ...")
    docs = []
    short_documents = 0
    doc_ids = []
    label_count = {}
    classes = {}
    for filepath in filepaths:
        # create the document ID
        label = os.path.basename(os.path.dirname(filepath).replace(" ", "_"))
        doc_id = os.path.splitext(os.path.basename(filepath))[0]
        if not doc_id.startswith(label):
            doc_id = "%s_%s" % (label, doc_id)
        # read body text
        log.debug("Reading text from %s ..." % filepath)
        body = read_text(filepath)
        if len(body) < options.min_doc_length:
            short_documents += 1
            continue
        docs.append(body)
        doc_ids.append(doc_id)
        if label not in classes:
            classes[label] = set()
            label_count[label] = 0
        classes[label].add(doc_id)
        label_count[label] += 1
    log.info("Kept %d documents. Skipped %d documents with length < %d" % (
    len(docs), short_documents, options.min_doc_length))
    if len(classes) < 2:
        log.warning("No ground truth available")
        classes = None
    else:
        log.info("Ground truth: %d classes - %s" % (len(classes), label_count))

    # Convert the documents in TF-IDF vectors and filter stopwords
    if options.stoplist_file is None:
        stopwords = load_stopwords("text/stopwords.txt")
    elif options.stoplist_file.lower() == "none":
        log.info("Using no stopwords")
        stopwords = set()
    else:
        log.info("Using custom stopwords from %s" % options.stoplist_file)
        stopwords = load_stopwords(options.stoplist_file)
    log.info(
        "Pre-processing data (%d stopwords, tfidf=%s, normalize=%s, min_df=%d) ..." % (
        len(stopwords), options.apply_tfidf, options.apply_norm,
        options.min_df))
    (X, terms, tfidf) = preprocess(
        docs, stopwords, min_df=options.min_df, apply_tfidf=options.apply_tfidf,
        apply_norm=options.apply_norm)
    log.info("Built document-term matrix: %d documents, %d terms" % (
        X.shape[0], X.shape[1]))

    # Store the corpus
    prefix = options.prefix
    if prefix is None:
        prefix = "corpus"
    log.info("Saving corpus '%s'" % prefix)
    save_corpus(prefix, X, terms, doc_ids, classes)
    save_tfidf(prefix, tfidf)


# --------------------------------------------------------------

if __name__ == "__main__":
    main()
