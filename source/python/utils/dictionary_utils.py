
__author__ = 'fpena'


def sort_dictionary_keys(dictionary):
    keys = []
    for key in sorted(dictionary, key=dictionary.get, reverse=True):
        keys.append(key)

    return keys

# dict = {'U1': 1.5, 'U2': 1.2, 'U3': 1.8, 'U4': 0.3, 'U5': 0.6}
# print(sort_dictionary_keys(dict))
