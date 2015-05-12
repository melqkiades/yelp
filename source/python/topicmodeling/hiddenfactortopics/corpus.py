import re
from topicmodeling.hiddenfactortopics.vote import Vote

__author__ = 'fpena'


class Corpus:

    WHITESPACE_SEPARATOR_REGEX = r"\S+"

    def __init__(self):

        self._num_users = 0
        self._num_items = 0
        self._num_words = 0

        self._user_count_map = {}
        self._item_count_map = {}
        self._vote_list = []
        self._user_id_map = {}
        self._item_id_map = {}
        self._word_count_map = {}
        self._word_id_map = {}

    def load_data(self, file_name, max_lines):
        self.process_reviews(file_name, max_lines)
        self.build_votes(file_name, max_lines)

    # OK, returns the same results as the McAuley implementation
    def process_reviews(self, file_name, max_lines):
        """
        Processes all the reviews contained in the given file and builds three
        objects in this class: the number of reviews per user, the number of
        reviews per item and the frequency of every word that appears in the
        reviews

        :param file_name: the file that contains the ratings and reviews
        :param max_lines: this indicates the maximum number of lines in the file
        to be processed by the method
        """
        num_lines_read = 0

        with open(file_name, 'r') as votes_file:
            for line in votes_file:
                # print(line)
                # We split the line by white spaces
                iterator = re.finditer(self.WHITESPACE_SEPARATOR_REGEX, line)
                user_name = next(iterator).group(0)
                item_name = next(iterator).group(0)
                rating = next(iterator).group(0)
                vote_date = next(iterator).group(0)
                num_words = next(iterator).group(0)

                # print("user =", user_name,
                #       "\titem =", item_name,
                #       "\trating =", rating,
                #       "\ttime =", vote_date,
                #       "\twords =", num_words)

                for token in iterator:
                    word = token.group(0)
                    if word not in self._word_count_map:
                        self._word_count_map[word] = 0
                    self._word_count_map[word] += 1
                    # print(word)

                if user_name not in self._user_count_map:
                    self._user_count_map[user_name] = 0
                if item_name not in self._item_count_map:
                    self._item_count_map[item_name] = 0

                self._user_count_map[user_name] += 1
                self._item_count_map[item_name] += 1

                num_lines_read += 1

                if 0 < max_lines <= num_lines_read:
                    break

            print("\nnUsers =", len(self._user_count_map),
                  "nItems =", len(self._item_count_map),
                  "nRatings =", num_lines_read)

    # Returns different results each time because elements are stored in a map,
    # and when ordering the words by frequency, it is unpredictable which word
    # is going to appear from a set of words that have the same frequency
    def build_votes(self, file_name, max_lines):

        # print("***************************")
        # print("***************************")
        # print("***************************")
        # print("***************************\n")

        self._num_users = 0
        self._num_items = 0

        min_users = 0
        min_items = 0
        max_words = 5000

        word_id_map = Corpus.build_word_id_map(self._word_count_map, max_words)
        self._num_words = len(word_id_map)
        # print(word_id_map)

        num_lines_read = 0

        with open(file_name, 'r') as votes_file:
            for line in votes_file:
                vote = Vote()

                # We split the line by white spaces
                iterator = re.finditer(self.WHITESPACE_SEPARATOR_REGEX, line)
                user_name = next(iterator).group(0)
                item_name = next(iterator).group(0)
                rating = float(next(iterator).group(0))
                vote_date = next(iterator).group(0)
                num_words = next(iterator).group(0)

                for token in iterator:
                    word = token.group(0)
                    if word in word_id_map:
                        vote.word_list.append(word_id_map[word])

                if self._user_count_map[user_name] >= min_users:
                    if user_name not in self._user_id_map:
                        self._user_id_map[user_name] = self.num_users
                        self._num_users += 1
                    vote.user = self._user_id_map[user_name]
                else:
                    vote.user = 0

                if self._item_count_map[item_name] >= min_items:
                    if item_name not in self._item_id_map:
                        self._item_id_map[item_name] = self.num_items
                        self._num_items += 1
                    vote.item = self._item_id_map[item_name]
                else:
                    vote.item = 0

                vote.rating = rating
                vote.date = vote_date
                self.vote_list.append(vote)

                # print("\nvoteUser = %d, voteItem = %d, voteRating = %f, numWords = %d", vote.user, vote.item, vote.rating, len(vote.word_list))
                # print("voteUser = ", vote.user,
                #       "\tvoteItem = ", vote.item,
                #       "\tvoteRating = ", vote.rating,
                #       "\tnumWords = ", len(vote.word_list))
                # print("voteUser =", vote.user,
                #       "\tvoteItem =", vote.item,
                #       "\tvoteRating =", vote.rating,
                #       "\tnumWords =", len(vote.word_list),
                #       "******",
                #       "\tactualUser =", user_name,
                #       "\tactualItem =", item_name,
                #       "\tactualWords =", num_words)


                num_lines_read += 1

                if 0 < max_lines <= num_lines_read:
                    break

    @staticmethod
    def build_word_id_map(word_count, max_words):
        """
        Based on the given word count, builds a dictionary that assigns an ID
        to each word (they key is the word itself, and the value is the ID),
        where the most frequent word in all the reviews has ID = 1, the second
        most frequent has ID = 2, and so on.

        :param word_count: a dictionary where the key is the word itself and the
        value is the number of times that that word appears in all reviews
        :param max_words: a parameter that indicates how many words should be
        returned. It just tell the method to return the top-n most frequent
        words (n = max_words)
        :return: a dictionary, where the key is the word and the value is the ID
        of the word
        """

        # print(word_count)

        word_list = sorted(word_count, key=word_count.get, reverse=True)

        if len(word_list) < max_words:
            max_words = len(word_list)

        word_id_map = {}

        for index in range(max_words):
            word_id_map[word_list[index]] = index

        return word_id_map

    @property
    def vote_list(self):
        """
        :rtype: list[Vote]
        """
        return self._vote_list

    @vote_list.setter
    def vote_list(self, value):
        """
        :type value: list[Vote]
        """
        self.vote_list = value

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def num_words(self):
        return self._num_words



# my_file = '/Users/fpena/tmp/SharedFolder/code_RecSys13/Arts-short.votes'
# my_corpus = Corpus(my_file, 0)
# my_corpus.process_reviews(my_file, 0)
# print(my_corpus._word_count_map)
# my_corpus.build_votes(my_file, 0)


# my_text = "A3MJRT2OEQX7HN B0009VEM4U 3.0 1361577600 23 great snippers but they are much bigger and bulkier than i expected ill keep them but not for the reason i purchased them"
# my_iterator = re.finditer(Corpus.WHITESPACE_SEPARATOR_REGEX, my_text)

# for token in my_iterator:
#     print(token.group(0))

x = {'a': 2, 'b': 4, 'c': 3, 'd': 1, 'e': 0}
# sorted_x = sorted(x, key=x.get, reverse=True)

# print(Corpus.build_word_id_map(x, 2))

# for element in sorted_x:
#     print(element)
