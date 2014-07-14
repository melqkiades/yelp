from etl import ETLUtils
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity import fourcity_clusterer

__author__ = 'fpena'


class CluOverall:

    def __init__(self, reviews, is_collaborative=False):
        self.reviews = reviews
        self.is_collaborative = is_collaborative
        self.user_dictionary = extractor.initialize_users(self.reviews)
        self.user_cluster_dictionary = fourcity_clusterer.build_user_clusters(self.reviews)
        pass

    def predict_rating(self, user_id, hotel_id):

        if user_id not in self.user_dictionary:
            return None

        cluster_name = self.user_dictionary[user_id].cluster

        # We remove the given user from the cluster in order to avoid bias
        cluster_users = list(self.user_cluster_dictionary[cluster_name])

        if user_id not in cluster_users:
            print(user_id)
            print(cluster_name)

        cluster_users.remove(user_id)

        filtered_reviews = ETLUtils.filter_records(self.reviews, 'offering_id',
                                                   [hotel_id])
        filtered_reviews = ETLUtils.filter_out_records(filtered_reviews, 'user_id',
                                                       [user_id])

        ratings_sum = 0
        ratings_count = 0
        for user in cluster_users:

            user_reviews = ETLUtils.filter_records(filtered_reviews, 'user_id',
                                                   [user])

            for review in user_reviews:
                ratings_sum += review['overall_rating']
                ratings_count += 1

        predicted_rating = None
        if ratings_count > 0:
            # predicted_rating = 3.9
            predicted_rating = float(ratings_sum) / float(ratings_count)

        return predicted_rating