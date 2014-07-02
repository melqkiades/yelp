import time
import operator
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity import fourcity_clusterer

__author__ = 'fpena'


class CluCFEuc:

    def __init__(self):
        self.reviews = extractor.pre_process_reviews()
        self.user_dictionary = extractor.initialize_users(self.reviews, 10)
        self.user_cluster_dictionary = fourcity_clusterer.build_user_clusters(self.reviews)
        self.users = extractor.get_user_list(self.reviews, 10)
        self.items = extractor.get_item_list(self.reviews, 2)
        self.user_reviews_dictionary = fourcity_clusterer.build_user_reviews_dictionary(self.reviews, self.users)

    def clu_cf_euc(self, reviews, user_id, hotel_id):

        average_overall_rating = self.user_dictionary[
            user_id].average_overall_rating

        cluster_name = self.user_dictionary[user_id].cluster

        # We remove the given user from the cluster in order to avoid bias
        cluster_users = list(self.user_cluster_dictionary[cluster_name])
        cluster_users.remove(user_id)

        similarities_sum = 0.
        similarities_ratings_sum = 0.
        for cluster_user in cluster_users:
            cluster_user_overall_rating = self.user_dictionary[cluster_user].average_overall_rating
            users_similarity = self.calculate_users_similarity(cluster_user, user_id)
            # users_similarity = 0
            user_item_rating = fourcity_clusterer.get_user_item_overall_rating(
                self.user_reviews_dictionary[user_id], user_id, hotel_id)
            # user_item_rating = 4

            if user_item_rating is not None:
                similarities_sum += users_similarity
                similarities_ratings_sum += users_similarity * (user_item_rating - cluster_user_overall_rating)
        predicted_rating = None
        error = None

        if similarities_sum > 0:
            predicted_rating = \
                average_overall_rating + similarities_ratings_sum / similarities_sum
            error = abs(predicted_rating - average_overall_rating)
            # print('Predicted rating: %f' % predicted_rating)

        # print('Average overall rating: %f' % average_overall_rating)

        return predicted_rating, error

    def clu_cf_euc_list(self):
        predicted_ratings = []
        errors = []

        index = 0
        print('CluCFEuc')
        print('Total reviews: %i' % len(self.reviews))

        for review in self.reviews:
            print('Index: %i' % index)
            index += 1

            predicted_rating, error = self.clu_cf_euc(
                self.reviews, review['user_id'],
                review['offering_id'])
            # predicted_rating = 4
            # error = 0
            predicted_ratings.append(predicted_rating)
            errors.append(error)

        return predicted_ratings, errors

    def clu_cf_euc_items(self):
        predicted_ratings = {}
        errors = []

        index = 0
        print('CluCFEuc Items')
        print('Total reviews: %i' % len(self.reviews))

        for user in self.users:
            print('Index: %i' % index)
            index += 1
            predicted_ratings[user] = {}

            five_star_hotels = extractor.get_five_star_hotels_from_user(self.user_reviews_dictionary[user])
            print(five_star_hotels)

            for item in self.items:
                predicted_rating, _ = self.clu_cf_euc(self.reviews, user, item)
                # predicted_ratings[user][item] = self.clu_cf_euc(self.reviews, user, item)
                if predicted_rating is not None:
                    predicted_ratings[user][item] = predicted_rating

            print(predicted_ratings[user])


        print(predicted_ratings)
        return predicted_ratings, errors

    def calculate_users_similarity(self, user_id1, user_id2):
        """
        Calculates the similarity between two users based on how similar are their
        ratings in the reviews

        :param user_id1: the ID of user 1
        :param user_id2: the ID of user 2
        :return: a float with the similarity between the two users. Since this
        function is based on euclidean distance to calculate the similarity, a
        similarity of 0 indicates that the users share exactly the same tastes
        """
        user_weights1 = self.user_dictionary[user_id1].criteria_weights
        user_weights2 = self.user_dictionary[user_id2].criteria_weights

        return fourcity_clusterer.calculate_euclidean_distance(user_weights1, user_weights2)

    def take_top_n(self, predicted_ratings, n):

        sorted_ratings = sorted(predicted_ratings.iteritems(), key=operator.itemgetter(1))
        sorted_ratings.reverse()

        return sorted_ratings[:n]




def main():
    clusterer = CluCFEuc()
    _, errors = clusterer.clu_cf_euc_list()
    # _, errors = clusterer.clu_cf_euc_items()
    mean_absolute_error = fourcity_clusterer.calculate_mean_average_error(errors)
    print('Mean Absolute error: %f' % mean_absolute_error)
    root_mean_square_error = fourcity_clusterer.calculate_root_mean_square_error(errors)
    print('Root mean square error: %f' % root_mean_square_error)


start_time = time.time()
main()
end_time = time.time() - start_time
print("--- %s seconds ---" % end_time)