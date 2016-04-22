
import colorsys
import time

import nltk
import re
from pylatex.base_classes import CommandBase, Arguments
from pylatex import Document, Section, Subsection, UnsafeCommand, Tabular
from pylatex.utils import italic
from pylatex.package import Package

from topicmodeling.context.topic_model_analyzer import load_topic_model, \
    split_topic
from utils.constants import Constants
import sys
reload(sys)
sys.setdefaultencoding('utf8')


class ColorBoxCommand(CommandBase):
    """
    A class representing a custom LaTeX command.

    This class represents a custom LaTeX command named
    ``exampleCommand``.
    """

    _latex_name = 'exampleCommand'
    packages = [Package('color')]


def extract_words(text):
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

    review_words = []

    for sentence in sentences:
        # words = [word.strip(string.punctuation) for word in sentence.split()]
        words = [word.lower() for word in nltk.word_tokenize(sentence)]
        review_words.extend(words)

    return review_words


def extract_topic_words(topic_model, topic_ids):

    topic_words_map = {}
    for topic_id in topic_ids:
        probability_words = topic_model.print_topic(topic_id, topn=10).split(' + ')
        words = set([word.split('*')[1] for word in probability_words])
        topic_words_map[topic_id] = words

    return topic_words_map


def bold_mapper(text):
    return '\\textbf{%s}' % text


def background_color_mapper(text_color):
    text_color_split = text_color.split('|||')
    text = text_color_split[0]
    red = float(text_color_split[1])
    green = float(text_color_split[2])
    blue = float(text_color_split[3])
    return '\\colorbox[rgb]{%f,%f,%f}{%s}' % (red, green, blue, text)


class TopicLatexGenerator:

    def __init__(self, lda_based_context):
        self.lda_based_context = lda_based_context
        self.doc = Document(Constants.ITEM_TYPE + '-topic-models')
        self.num_cols = 5
        self.num_topics = self.lda_based_context.num_topics
        self.rgb_tuples = None
        self.automatic_context_topic_colors = None
        self.manual_context_topic_colors = None
        self.automatic_context_topic_ids = None
        self.manual_context_topic_ids = None
        self.automatic_context_topic_words = None
        self.manual_context_topic_words = None
        self.headers = None
        self.topic_words_map = None
        self.table_format = '|' + 'c|' * (self.num_cols + 1)
        self.init_colors()
        self.init_headers()
        self.init_topic_words()
        self.init_topic_ids()
        new_comm = UnsafeCommand('newcommand', '\exampleCommand', options=4,
            extra_arguments=r'\colorbox[rgb]{#1,#2,#3}{#4} \color{black}')
        self.doc.append(new_comm)
        new_comm2 = UnsafeCommand('tiny')
        self.doc.append(new_comm2)

    def init_colors(self):
        golden_ratio = 0.618033988749895
        hsv_tuples = [((x * golden_ratio) % 1.0, 0.5, 0.95)
                      for x in range(self.num_topics)]
        self.rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)

        color_index = 0
        self.automatic_context_topic_colors = {}
        for topic in self.lda_based_context.context_rich_topics:
            topic_id = topic[0]
            self.automatic_context_topic_colors[topic_id] = \
                self.rgb_tuples[color_index]
            color_index += 1

        color_index = 0
        self.manual_context_topic_colors = {}
        for topic_id in range(self.num_topics):
            topic_score = split_topic(
                self.lda_based_context.topic_model.print_topic(
                    topic_id, topn=self.num_cols))
            if topic_score['score'] > 0:
                self.manual_context_topic_colors[topic_id] = color_index
                color_index += 1

    def init_headers(self):
        self.headers = ['ID']
        for column_index in range(self.num_cols):
            self.headers.append('Word ' + str(column_index + 1))

    def init_topic_words(self):
        self.topic_words_map = \
            extract_topic_words(
                self.lda_based_context.topic_model, range(self.num_topics))

    def init_topic_ids(self):
        self.automatic_context_topic_ids = [
            topic[0] for topic in self.lda_based_context.context_rich_topics]

        self.manual_context_topic_ids = []
        for topic_id in range(self.num_topics):
            topic_score = split_topic(
                self.lda_based_context.topic_model.print_topic(
                    topic_id, topn=self.num_cols))
            if topic_score['score'] > 0:
                self.manual_context_topic_ids.append(topic_id)

    def create_automatic_context_topics(self):

        with self.doc.create(Section('Context-rich topic models (automatic)')):
            num_context_topics = len(self.lda_based_context.context_rich_topics)

            with self.doc.create(Tabular(self.table_format)) as table:
                table.add_hline()
                table.add_row(self.headers, mapper=bold_mapper)
                table.add_hline()

                for topic in self.lda_based_context.context_rich_topics:
                    topic_id = topic[0]
                    row = [str(topic_id + 1)]
                    topic_words =\
                        self.lda_based_context.topic_model.print_topic(
                            topic_id, topn=self.num_cols).split(' + ')
                    row.extend(topic_words)
                    table.add_row(row)

                table.add_hline()

            self.doc.append(UnsafeCommand('par'))
            self.doc.append(
                'Number of context-rich topics: %d' % num_context_topics)

    def create_manual_context_topics(self):
        with self.doc.create(Section('Context-rich topic models (manual)')):

            num_context_topics = 0

            with self.doc.create(Tabular(self.table_format)) as table:
                table.add_hline()
                table.add_row(self.headers, mapper=bold_mapper)
                table.add_hline()

                for topic_id in range(self.num_topics):
                    topic_score = split_topic(
                        self.lda_based_context.topic_model.print_topic(
                            topic_id, topn=self.num_cols))
                    if topic_score['score'] > 0:
                        color_id = self.manual_context_topic_colors[topic_id]
                        color = self.rgb_tuples[color_id]
                        id_cell = str(topic_id)+str('|||')+str(color[0]) + \
                            '|||'+str(color[1])+'|||'+str(color[2])
                        row = [id_cell]
                        for column_index in range(self.num_cols):
                            word = topic_score['word' + str(column_index)]
                            word_color = word+str('|||')+str(color[0])+'|||' + \
                                str(color[1])+'|||'+str(color[2])
                            row.append(word_color)
                        table.add_row(row, mapper=background_color_mapper)

                        num_context_topics += 1

                table.add_hline()

            self.doc.append(UnsafeCommand('par'))
            self.doc.append(
                'Number of context-rich topics: %d' % num_context_topics)

    def create_reviews(self):
        with self.doc.create(Section('Reviews')):
            with self.doc.create(Subsection('A subsection')):

                review_index = 0
                for record in self.lda_based_context.records[:100]:
                    with self.doc.create(Subsection(
                                    'Review %d (%s)' % (
                            (review_index + 1), record[
                                Constants.PREDICTED_CLASS_FIELD]))):
                        for doc_part in self.build_text(
                                record[Constants.TEXT_FIELD]):
                            self.doc.append(doc_part)
                    review_index += 1

                # another_text = UnsafeCommand(
                #     arguments=Arguments(1.0, 0.5, 0.5, ' Hola '))
                #
                # self.doc.append('Also some crazy characters: $&#{}')
                # self.doc.append(' and some other things')
                # for color in self.rgb_tuples:
                #     red = color[0]
                #     green = color[1]
                #     blue = color[2]
                #     self.doc.append(
                #         ExampleCommand(
                #             arguments=Arguments(red, green, blue,
                #                                 'Hello World!')))

    def generate_pdf(self):
        self.create_automatic_context_topics()
        self.create_manual_context_topics()
        self.create_reviews()
        self.doc.generate_pdf()
        self.doc.generate_tex()

    def build_text(self, review):
        words = extract_words(review)
        doc_parts = []
        new_words = []
        for word in words:
            word_found = False
            for topic_id in self.manual_context_topic_ids:
                if word in self.topic_words_map[topic_id]:
                    doc_parts.append(' '.join(new_words))
                    # doc_parts.append('topic: %d word: %s' % (topic_id, word))
                    color_id = self.manual_context_topic_colors[topic_id]
                    color = self.rgb_tuples[color_id]
                    doc_parts.append(ColorBoxCommand(
                        arguments=Arguments(
                            color[0], color[1], color[2], word)))
                    new_words = []
                    word_found = True
                    break
            if not word_found:
                new_words.append(word)
        doc_parts.append(' '.join(new_words))

        return doc_parts



def fill_topics_document(doc, lda_based_context):
    new_comm = UnsafeCommand('newcommand', '\exampleCommand', options=4,
        extra_arguments=r'\colorbox[rgb]{#1,#2,#3}{#4} \color{black}')
    doc.append(new_comm)

    num_cols = 5
    N = 150
    golden_ratio = 0.618033988749895
    hsv_tuples = [((x * golden_ratio) % 1.0, 0.5, 0.95) for x in range(N)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    automatic_context_topic_colors = {}
    manual_context_topic_colors = {}

    new_comm2 = UnsafeCommand('tiny')
    doc.append(new_comm2)

    table_format = '|' + 'c|' * (num_cols + 1)

    with doc.create(Section('Context-rich topic models (automatic)')):

        num_context_topics = len(lda_based_context.context_rich_topics)

        with doc.create(Tabular(table_format)) as table:
            table.add_hline()

            headers = ['ID']
            for column_index in range(num_cols):
                headers.append('Word ' + str(column_index + 1))
            table.add_row(headers)
            table.add_hline()

            color_index = 0
            for topic in lda_based_context.context_rich_topics:
                topic_id = topic[0]
                row = [str(topic_id + 1)]
                topic_words = lda_based_context.topic_model.print_topic(topic_id, topn=num_cols).split(' + ')
                row.extend(topic_words)
                table.add_row(row)
                automatic_context_topic_colors[topic_id] =\
                    rgb_tuples[color_index]
                color_index += 1

            table.add_hline()

        doc.append(UnsafeCommand('par'))
        doc.append('Number of context-rich topics: %d' % num_context_topics)

    with doc.create(Section('Context-rich topic models (manual)')):

        num_context_topics = 0

        with doc.create(Tabular(table_format)) as table:
            table.add_hline()
            headers = ['ID']
            for column_index in range(num_cols):
                headers.append('Word ' + str(column_index + 1))
            table.add_row(headers)
            table.add_hline()

            color_index = 0
            for topic_id in range(lda_based_context.num_topics):
                topic_score = split_topic(
                    lda_based_context.topic_model.print_topic(topic_id, topn=num_cols))
                if topic_score['score'] > 0:
                    row = [topic_id]
                    for column_index in range(num_cols):
                        row.append(topic_score['word' + str(column_index)])
                    table.add_row(row)
                    manual_context_topic_colors[topic_id] = color_index

                num_context_topics += 1

            table.add_hline()

        doc.append(UnsafeCommand('par'))
        doc.append('Number of context-rich topics: %d' % num_context_topics)

    with doc.create(Section('Reviewseeeeeee!!!!!')):
        with doc.create(Subsection('A subsection')):

            another_text = UnsafeCommand(
                arguments=Arguments(1.0, 0.5, 0.5, ' Hola '))

            doc.append('Also some crazy characters: $&#{}')
            doc.append(' and some other things')
            for color in rgb_tuples:
                red = color[0]
                green = color[1]
                blue = color[2]
                doc.append(
                    ColorBoxCommand(
                        arguments=Arguments(red, green, blue, 'Hello World!')))

        review_index = 0
        for record in lda_based_context.records[:100]:
            print(review_index)
            with doc.create(Subsection('Review %d (%s)' %
                ((review_index + 1), record[Constants.PREDICTED_CLASS_FIELD]))):
                doc.append(record[Constants.TEXT_FIELD])
            review_index += 1


def fill_document(doc):
    """Add a section, a subsection and some text to the document.
    :param doc: the document
    :type doc: :class:`pylatex.document.Document` instance
    """

    new_comm = UnsafeCommand('newcommand', '\exampleCommand', options=4,
                             extra_arguments=r'\colorbox[rgb]{#1,#2,#3}{#4} \color{black}')
    doc.append(new_comm)

    with doc.create(Section('A section')):
        doc.append('Some regular text and some ')
        doc.append(italic('italic text. '))

        with doc.create(Subsection('A subsection')):
            doc.append('Also some crazy characters: $&#{}')

    with doc.create(Section('The second section')):
        doc.append('Text for the second section ')
        doc.append(
            ColorBoxCommand(arguments=Arguments(1.0, 0.5, 0.5, 'Hello World!')))




def main():
    # Basic document
    # doc = Document('basic')
    # fill_document(doc)
    #
    # doc.generate_pdf()
    # doc.generate_tex()

    # doc = Document(Constants.ITEM_TYPE + '-topic-models')
    lda_based_context = load_topic_model(0, 0)
    # lda_based_context = None
    # fill_topics_document(doc, lda_based_context)

    # doc.generate_pdf()
    # doc.generate_tex()
    topic_latex_generator = TopicLatexGenerator(lda_based_context)
    topic_latex_generator.generate_pdf()

    # Document with `\maketitle` command activated
    # doc = Document()


    # doc.preamble.append(Command('title', 'Awesome Title'))
    # doc.preamble.append(Command('author', 'Anonymous author'))
    # doc.preamble.append(Command('date', NoEscape(r'\today')))
    # doc.append(NoEscape(r'\maketitle'))
    #
    # # fill_document(doc)
    #
    # doc.generate_pdf('basic_maketitle', clean=False)
    #
    # # Add stuff to the document
    # # with doc.create(Section('A second section')):
    # #     doc.append('Some text.')
    #
    # doc.generate_pdf('basic_maketitle2')
    # tex = doc.dumps()  # The document as string in LaTeX syntax

start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)

# print(background_color_mapper('hola', (0.5, 0.7, 0.2)))


def build_text(review, topic_words_map):
    words = extract_words(review)
    doc_parts = []
    new_words = []
    for word in words:
        word_found = False
        for topic_id in topic_words_map.keys():
            if word in topic_words_map[topic_id]:
                doc_parts.append(' '.join(new_words))
                doc_parts.append('topic: %d word: %s' % (topic_id, word))
                new_words = []
                word_found = True
                break
        if not word_found:
            new_words.append(word)
    doc_parts.append(''.join(new_words))

    return doc_parts





# lda_based_context = load_topic_model(0, 0)
# probability_words = lda_based_context.topic_model.print_topic(0, topn=10).split(' + ')
# for probability_word in probability_words:
#     print(probability_word.split('*')[1])
# review = lda_based_context.records[0][Constants.TEXT_FIELD]
# print(review)
# topic_words_map = extract_topic_words(lda_based_context.topic_model, range(10))
# print(detect_word_in_topic(topic_words_map, 'summer'))
# print(extract_words(lda_based_context.records[0][Constants.TEXT_FIELD]))
# print(build_text(lda_based_context.records[0][Constants.TEXT_FIELD], topic_words_map))


# print(lda_based_context.topic_model.get_topic_terms(0))


