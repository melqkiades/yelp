from etl import ETLUtils
from tripadvisor.fourcity import extractor
from tripadvisor.fourcity import fourcity_clusterer

__author__ = 'fpena'


class CluOverall:

    def __init__(
            self, reviews, significant_criteria_range=None,
            is_collaborative=False, similarity_metric='euclidean'):
        self.reviews = reviews
        self.is_collaborative = is_collaborative
        self.user_dictionary = extractor.initialize_users(self.reviews)
        self.user_cluster_dictionary = fourcity_clusterer.build_user_clusters(self.reviews)
        self.user_ids = extractor.get_groupby_list(self.reviews, 'user_id')
        self.user_similarity_matrix =\
            fourcity_clusterer.build_user_similarities_matrix(
                self.user_ids, self.user_dictionary, similarity_metric)

    def predict_rating(self, user_id, item_id):

        if user_id not in self.user_dictionary:
            return None

        cluster_name = self.user_dictionary[user_id].cluster

        # We remove the given user from the cluster in order to avoid bias
        cluster_users = list(self.user_cluster_dictionary[cluster_name])
        cluster_users.remove(user_id)

        similarities_sum = 0.
        similarities_ratings_sum = 0.
        num_users = 0
        for cluster_user in cluster_users:
            users_similarity = self.user_similarity_matrix[cluster_user][user_id]

            if item_id in self.user_dictionary[cluster_user].item_ratings:
                cluster_user_item_rating = self.user_dictionary[cluster_user].item_ratings[item_id]
                similarities_sum += users_similarity

                if self.is_collaborative:
                    similarities_ratings_sum += users_similarity * cluster_user_item_rating
                else:
                    similarities_ratings_sum += cluster_user_item_rating
                num_users += 1

        predicted_rating = None
        if num_users > 0:
            if self.is_collaborative:
                predicted_rating = float(similarities_ratings_sum) / float(similarities_sum)
            else:
                predicted_rating = float(similarities_ratings_sum) / float(num_users)

            if predicted_rating > 5:
                predicted_rating = 5
            elif predicted_rating < 1:
                predicted_rating = 1

        return predicted_rating