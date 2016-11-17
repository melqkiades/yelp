import random

import numpy

from utils.constants import Constants


grouped_hotel_context_words = {
    'airport': {'airport', 'shuttle', 'plane', 'flight', 'transportation',
                'bus'},
    'holiday': {'holiday', 'vacation', 'staycation', 'getaway', '@', '@@'},
    'conference': {'conference', 'convention', 'group', 'meeting', 'attended',
                   '@'},
    'pets': {'dog', 'pet', 'cat', '@', '@@', '@@@'},
    'discount': {'discount', 'hotwire', 'groupon', 'deal', 'priceline', '@'},
    'wedding': {'wedding', 'reception', 'ceremony', 'marriage', '@', '@@'},
    'festivities': {'christmas', 'thanksgiving', 'holiday', '@', '@@', '@@@'},
    'family': {'mom', 'dad', 'father', 'mother', 'grandma', 'grandmother',
               'grandpa', 'grandfather', 'parent', 'grandparent',
               'daughter', 'uncle', 'sister', 'brother', 'aunt', 'sibling',
               'child',  'daughter', 'son', 'kid', 'boy', 'girl', 'family'},
    'romantic': {'date', 'anniversary', 'romantic', 'girlfriend',
                 'boyfriend', 'bf', 'gf', 'hubby', 'husband', 'wife',
                 'fiance', 'fiancee', 'weekend', 'getaway', 'romance'},
    'anniversary': {'husband', 'wife', 'weekend', 'anniversary', 'hubby', '@'},
    'gambling': {'gamble', 'casino', 'slot', 'poker' 'roulette', '@'},
    'party': {'party', 'friend', 'music', 'group', 'nightlife', 'dj'},
    'business': {'business', 'work', 'job', 'colleague', 'coworker', '@'},
    'parking': {'car', 'parking', 'valet', 'driver', '@', '@@'},
    'season': {'winter', 'spring', 'summer', 'fall', 'autumn', '@'},
    'month': {'january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november',
              'december'},
    'day': {'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'weekday', 'weekend'},
    'occasion': {'birthday', 'anniversary', 'celebration', 'date',
                 'wedding', 'honeymoon'},
    'sport_event': {'football', 'baseball', 'basketball', 'game', 'field',
                    'match', 'tournament', 'ticket'},
    'outdoor': {'golf', 'tenni', 'court', 'field', 'horse', 'cabana'
                'training', 'exercise', 'bike', 'cycle', 'kart', 'cart',
                'fitness'},
    'relax': {
        'relax', 'quiet', 'getaway', 'stress', 'relief', 'massage',
        'spa', 'steam', 'jacuzzi', 'facial', 'treatment', 'relaxing',
        'treatment'
    },
    'accessibility': {
        'wheelchair', 'handicap', 'ramp', '@', '@@', '@@@'
    },
    # 'non-contextual': {'room'}
}

grouped_restaurant_context_words = {
    'breakfast': {'brunch', 'breakfast', 'morning', 'pancakes', 'omelette',
                  'waffle'},
    'lunch': {'afternoon', 'lunch', 'noon',  '@', '@@', '@@@'},
    'dinner': {'dinner', 'evening', 'night',  '@', '@@', '@@@'},
    'romantic': {'date', 'night', 'anniversary', 'romantic', 'girlfriend',
                 'boyfriend', 'bf', 'gf', 'hubby', 'husband', 'wife',
                 'fiance', 'fiancee', 'weekend'},
    'party': {'party', 'friend', 'music', 'group', 'disco',
              'club', 'guy', 'people', 'night', 'nightlife'},
    'kids': {'child', 'kid', 'boy', 'girl', 'family', '@'},
    'parking': {'parking', 'car', 'valet', 'driver', '@', '@@'},
    'work': {'busines', 'colleague', 'workplace', 'job', 'meeting', 'coworker',
             'office'},
    'family': {'mom', 'dad', 'father', 'mother', 'grandma', 'grandmother',
               'grandpa', 'grandfather', 'parent', 'grandparent',
               'daughter', 'uncle', 'sister', 'brother', 'aunt', 'sibling',
               'daughter', 'son'},
    'friends': {'friend', 'group', 'girl', 'boy', 'guy', '@'},
    'time': {'morning', 'noon', 'afternoon', 'evening', 'night', '@'},
    'birthday': {'birthday', 'celebration', 'event', '@', '@@', '@@@'},
    'discount': {'deal', 'coupon', 'groupon', 'discount', '@', '@@'},
    'takeaway': {'delivery', 'takeaway', 'drive', 'thru', 'takeout',
                 'deliver'},
    'sports': {'sports', 'match', 'game', 'tv', 'football',
               'baseball', 'basketball', 'nfl'},
    'karaoke': {'song', 'karaoke', 'music', '@', '@@', '@@@'},
    'outdoor': {'outdoor', 'patio', 'outside', 'summer', '@', '@@'},
    'day': {'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'weekday', 'weekend'},
    # 'season': {'winter', 'summer', 'fall', 'autumn'},
    'accessibility': {'wheelchair', 'handicap', 'ramp', '@', '@@', '@@@'},
    # 'non-contextual': {'food'}
}

context_words = {
    'yelp_hotel': grouped_hotel_context_words,
    'fourcity_hotel': grouped_hotel_context_words,
    'yelp_restaurant': grouped_restaurant_context_words
}

hotel_context_words = set()
for groups in grouped_hotel_context_words.values():
    hotel_context_words |= groups

restaurant_context_words = set()
for groups in grouped_restaurant_context_words.values():
    restaurant_context_words |= groups

all_context_words = {
    'yelp_hotel': hotel_context_words,
    'fourcity_hotel': hotel_context_words,
    'yelp_restaurant': restaurant_context_words
}


def plant_seeds():

    if Constants.RANDOM_SEED is not None:
        print('random seed: %d' % Constants.RANDOM_SEED)
        random.seed(Constants.RANDOM_SEED)
    if Constants.NUMPY_RANDOM_SEED is not None:
        print('numpy random seed: %d' % Constants.NUMPY_RANDOM_SEED)
        numpy.random.seed(Constants.NUMPY_RANDOM_SEED)
