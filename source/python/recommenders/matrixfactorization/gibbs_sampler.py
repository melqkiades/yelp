import random

__author__ = 'fpena'


colors = ['red', 'black']
symbols = ['faces', 'numbers']

probability_table = [
    # Probability that color is red given that symbols is face (position 0) or
    # symbol is number (position 1)
    [5./6, 17./31],

    # Probability that symbol is face given that color is red (position 0) or
    # color is black (position 1)
    [5./22, 1./15]
]

class GibbsSampler:

    def __init__(self):
        self.sample_list = []

    def sample(self):

        # Set the initial values for the variables
        values = [
            random.randint(0, 1),  # Color value
            random.randint(0, 1)   # Symbol value
        ]

        for i in xrange(3700000):
            # Select which variable to take (colors or card types)
            var_type = random.randint(0, 1)
            if var_type == 0:
                still_variable = 1
            else:
                still_variable = 0

            new_random_value = random.random()

            if new_random_value > probability_table[var_type][values[still_variable]]:
                values[var_type] = 1
            else:
                values[var_type] = 0
            if i % 100 == 0:
                self.store_variables(values[0], values[1])
            # self.store_variables(values[0], values[1])

        print('Length', len(self.sample_list))

    def store_variables(self, color, symbol):
        self.sample_list.append((colors[color], symbols[symbol]))
        # print_variables(color, symbol)

    def count_results(self):

        pairs = [
            ('red', 'faces'),
            ('red', 'numbers'),
            ('black', 'faces'),
            ('black', 'numbers')
        ]

        for pair in pairs:
            pair_count = 0
            for sample in self.sample_list:
                if sample == pair:
                    pair_count += 1

            print(pair, pair_count)


def print_variables(color, symbol):
    print(colors[color], symbols[symbol])


sampler = GibbsSampler()
sampler.sample()
sampler.count_results()


# print(random.random())


