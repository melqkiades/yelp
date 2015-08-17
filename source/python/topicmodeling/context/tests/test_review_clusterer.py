from collections import Counter
import math
import numpy
import numpy.testing as nptest
from topicmodeling.context import review_utils
from topicmodeling.context import reviews_clusterer
from topicmodeling.context.review import Review

__author__ = 'fpena'

from unittest import TestCase

empty_paragraph = ""
paragraph1 =\
    "Good morning Dr. Adams. The patient is waiting for you in room number 3."
review_text1 = "We had dinner there last night. The food was delicious. " \
               "Definitely, is the best restaurant in town."
review_text2 = "Small bar, good music, good beer, bad food"
review_text3 = "Great hotel in Central Phoenix for a stay-cation, but not " \
               "necessarily a place to stay out of town and without a car. " \
               "Not much around the area, and unless you're familiar with " \
               "downtown, I would rather have a guest stay in Old Town " \
               "Scottsdale, etc. BUT if you do stay here, it's awesome. Great" \
               " boutique rooms. Awesome pool that's happening in the summer." \
               " A GREAT rooftop patio bar, and a very very busy lobby with " \
               "Gallo Blanco attached. A great place to stay, but have a car!"
review_text4 = "I've stayed at an Aloft before.  They are pretty much all the" \
               " same.  My king room was clean, modern, and well maintained." \
               "  I went in the winter, but had it been summer, I probably " \
               "would have taken advantage of the pool and outdoor lounge " \
               "area.  Even the gym looked appealing, and that's saying a " \
               "lot.  The major downfall (and I say major because the whole " \
               "point of a hotel is to sleep) is that the bed pillows are so" \
               " incredibly uncomfortable.  It is almost impossible to fall " \
               "asleep.  Someone please get some decent pillows in there!"
review_text5 = "Price to performance is pretty stellar. For $110/night I " \
               "enjoyed a large room with a luxurious bathroom, balcony, and " \
               "cool retro furnishings. Service was decent, but the " \
               "W Scottsdale is better on this front. Breakfast at their " \
               "restaurant was basic but worked. Stylings from the lobby to " \
               "the furniture in your room is really well done, the hotel has" \
               " a great boutique-y vibe but without the pretentiousness. " \
               "The location is good, only minutes from downtown Scottsdale. " \
               "One snafu while I was there... I sent out some dry cleaning " \
               "that was supposed to be back by 5:30pm, didn't happen. It was" \
               " a stressful half hour before they found my stuff " \
               "(all cleaned, phew!)."
review_text6 = "When I was a kid I loved the TV show Hotel starring " \
               "James Brolin (before he became Mr. Streisand). I was " \
               "fascinated by hotels. So much that I wanted to live in a " \
               "hotel. Room service. Chocolate on the pillow. Beds that " \
               "vibrate for a quarter. Well, I didn't always say they were " \
               "nice hotels. While I don't live in a hotel, I practically " \
               "live at The Clarendon.\n\nThings I have done at " \
               "The Clarendon. And I can't list everything because my " \
               "mother reads my reviews...\n\nSlept in perhaps the most " \
               "comfortable bed in the world. \nAttended a rockin' Halloween " \
               "party dressed as, c'mon, guess...Yep...Hello Kitty.\nAttended" \
               " photo shoots.\nHad drinks by that glorious pool and " \
               "waterfall.\nWatched the sunset on the fabulous rooftop patio." \
               " \nBecome a regular (yep, they even know my name!) at " \
               "Gallo Blanco.\nHad dozens of extremely happy out of town " \
               "visitors stay here. \nEnjoyed the men's urinals shaped like " \
               "big red lips.\nDried my hands on that amazing Dyson hand " \
               "dryer (insert hands for 12 seconds and they're drier than a " \
               "Napa Brut)\nLeft my beloved bomber jacket at GB and retrieved" \
               " it the next day...honest, honest employees.\n\nGreat staff " \
               "from hotel to restaurant. Attentive. Friendly. Professional. " \
               "I want to move somewhere else just so I can go on vacation " \
               "and stay at The Clarendon. But. Wait. I don't have to. I can " \
               "simply cross the street from my pad and enjoy this fabulous " \
               "Phoenix gem. \n\nPM for the R rated version. Chow."
review_text7 = "Oooooh, I liked it, very much so!  Just sad that I didn't get" \
               " to try more of its myriad offerings, especially the pizza.  " \
               "We enjoyed a lovely brunch here on a Friday.  Cute little " \
               "place, lots of ambiance, good people watching (quite the " \
               "hopping place on this midsummer day), tasty food.  We had a " \
               "round of Bloody Marys, and those were nicely done, as well.  " \
               "A great spot to just hang and nosh and chitchat with friends."
review_text8 = "Scary things to me:\nParis Hilton has a career. \nSarah Palin." \
               " Really. She's like the new Anita Bryant. \nMy fathers' " \
               "overgrown eyebrows. Trim, daddy, trim!\nDroves of high " \
               "schoolers leaving school just as I'm trying to find " \
               "Barrio Cafe. \n\nSeriously. Like thousands of them. And I " \
               "couldn't find the damn restaurant. What was I gonna do? Roll " \
               "down the window (well, not roll down, seriously, who rolls " \
               "down windows anymore?) and holler out in my best lispy voice," \
               " \"Hey, squeeze me fellas, I'm going to a wine tasting at the" \
               " Barrio Cafe. Y'all know where that is?\" Since all teenagers" \
               " carry guns, yes, that scared me. \n\nFinally. I found it. " \
               "And entered. \n\nCute. Cozy. Middle of the day, so empty. " \
               "Great, vibrant art. And a very attentive and friendly staff. " \
               "Often times at this time of day, restaurants really drop the " \
               "ball. No ball dropping here.\nI had La Paloma margarita which" \
               " is made with a grapefruit soda. Think Squirt and Patron! And" \
               " after a couple of these babies I was in el bano squirting " \
               "out that Patron. Dee lish however. \n\nI ordered the " \
               "Enchiladas Del Mar. Enchiladas of the Sea for my non Spanish " \
               "speaking peeps. You know what? It was good. Not great. And " \
               "the presentation was somewhat messy. I couldn't find the crab" \
               " or the scallops (which were the size of the ends of q-tips " \
               "by the way). They were lost in the sauce and the cheese. But " \
               "I devoured it as the blue corn tortillas sang to me. My " \
               "friend had a chickent torta. Big and rustic. Like Janet Reno." \
               " She loved it! \n\nI will most definitely be back. Hopefully " \
               "I will not have to navigate through the plethora of pubescent" \
               " people and hopefully I will pick that Barrio menu item that" \
               " will blow my calcentinas off. \n\nOh. That's blow my socks " \
               "off. BTW. Adios for now mi yelpitas!"
review_text9 = "Beef gyros are always good here."


class TestContextUtils(TestCase):

    def test_count_sentences(self):

        expected_value = 1
        actual_value = reviews_clusterer.count_sentences(empty_paragraph)
        self.assertEqual(actual_value, expected_value)

        expected_value = 2
        actual_value = reviews_clusterer.count_sentences(paragraph1)
        self.assertEqual(actual_value, expected_value)

        expected_value = 3
        actual_value = reviews_clusterer.count_sentences(review_text1)
        self.assertEqual(actual_value, expected_value)

        expected_value = 7
        actual_value = reviews_clusterer.count_sentences(review_text3)
        self.assertEqual(actual_value, expected_value)

    def test_count_words(self):

        expected_value = 0
        actual_value = reviews_clusterer.count_words(empty_paragraph)
        self.assertEqual(actual_value, expected_value)

        expected_value = 14
        actual_value = reviews_clusterer.count_words(paragraph1)
        self.assertEqual(actual_value, expected_value)

        expected_value = 17
        actual_value = reviews_clusterer.count_words(review_text1)
        self.assertEqual(actual_value, expected_value)

        expected_value = 87
        actual_value = reviews_clusterer.count_words(review_text3)
        self.assertEqual(actual_value, expected_value)

    def test_count_verbs(self):

        expected_value = 0
        tagged_words = review_utils.tag_words(empty_paragraph)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = reviews_clusterer.count_verbs(counts)
        self.assertEqual(actual_value, expected_value)

        expected_value = 2
        tagged_words = review_utils.tag_words(paragraph1)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = reviews_clusterer.count_verbs(counts)
        self.assertEqual(actual_value, expected_value)

        expected_value = 3
        tagged_words = review_utils.tag_words(review_text1)
        counts = Counter(tag for word, tag in tagged_words)
        actual_value = reviews_clusterer.count_verbs(counts)
        self.assertEqual(actual_value, expected_value)

    def test_process_review(self):

        expected_value = numpy.array([math.log(2), 0, 0, 0, 1])
        actual_value =\
            reviews_clusterer.get_review_metrics(Review(empty_paragraph))
        nptest.assert_allclose(actual_value, expected_value)

    def test_split_list_by_labels(self):

        lst = ['a', 'b', 'c', 'd', 'e', 'f']
        labels = [2, 0, 0, 1, 1, 0]
        expected_matrix = [
            ['b', 'c', 'f'],
            ['d', 'e'],
            ['a']
        ]
        actual_matrix = reviews_clusterer.split_list_by_labels(lst, labels)

        self.assertItemsEqual(actual_matrix, expected_matrix)
