import numpy


class DunnCalculator(object):
    """
    Calculates Dunn index as presented in Krzystof Kryszczuk and Paul Hurley's
    paper .
    """
    def __init__(self):
        pass

    def evaluate(self, clustering, matrix):
        dmin = self.min_intracluster_distances(clustering, matrix)
        dmax = self.max_intercluster_distance(clustering, matrix)
        dunn_index = dmin / dmax
        return dunn_index

    @classmethod
    def min_intracluster_distances(cls, clustering, matrix):
        """
        Calculates d_min, the minimum internal distance.
        @param clustering: The clustering being checked.
        @param matrix: The condensed matrix containing all distances.
        @return: d_min's value
        """
        distances = []
        for c in clustering.clusters:
            try:
                distances.append(numpy.min(get_intra_cluster_distances(c, matrix)))
            except SingularClusterException:
                # If we work with a singular cluster, we add 0s so that no min
                # function fails. The convention for the distance of a cluster
                # with only one element will be 0 in this case.
                distances.append(0)
        return numpy.min(distances)

    @classmethod
    def max_intercluster_distance(cls, clustering, matrix):
        """
        Calculates d_max, the maximum inter clusters.
        @param clustering: The clustering being checked.
        @param matrix: The condensed matrix containing all distances.
        @return: d_max' value
        """
        distances = []
        for i in range(len(clustering.clusters)-1):
            for j in range(i+1, len(clustering.clusters)):
                distances.extend(get_inter_cluster_distances(
                    i, j, clustering.clusters, matrix))
        return numpy.max(distances)


def get_intra_cluster_distances(cluster, matrix):
    # TODO: graph.tools.cut can be a wrapper of this function
    distances = []
    cluster_elements = cluster.all_elements
    for i in range(len(cluster.all_elements)-1):
        for j in range(i+1, len(cluster.all_elements)):
            distances.append(matrix[cluster_elements[i], cluster_elements[j]])

    if len(distances) == 0:
        # This means the cluster has only one element. We raise an exception
        # so the caller has to define its behaviour
        raise SingularClusterException("The cluster has only one element")

    return distances


def get_inter_cluster_distances(i, j, clusters, matrix):
    distances = []
    for cluster_i_element in clusters[i].all_elements:
        for cluster_j_element in clusters[j].all_elements:
            distances.append(matrix[cluster_i_element, cluster_j_element])
    return distances





class SingularClusterException(Exception):
    pass


def fpena_evaluate(cluster_list, matrix):
    dmin = fpena_min_intracluster_distances(cluster_list, matrix)
    dmax = fpena_max_intercluster_distance(cluster_list, matrix)

    print('dmin: %f' % dmin)
    print('dmax: %f' % dmax)

    dunn_index = dmin / dmax
    return dunn_index


def fpena_min_intracluster_distances(cluster_list, matrix):
    """
    Calculates d_min, the minimum internal distance.
    @param clustering: The clustering being checked.
    @param matrix: The condensed matrix containing all distances.
    @return: d_min's value
    """
    distances = []
    for cluster in cluster_list.values():
        try:
            distances.append(numpy.min(fpena_get_intra_cluster_distances(
                cluster, matrix)))
        except SingularClusterException:
            # If we work with a singular cluster, we add 0s so that no min
            # function fails. The convention for the distance of a cluster
            # with only one element will be 0 in this case.
            distances.append(1)
    return numpy.min(distances)


def fpena_max_intercluster_distance(cluster_list, matrix):
    """
    Calculates d_max, the maximum inter clusters.
    @param clustering: The clustering being checked.
    @param matrix: The condensed matrix containing all distances.
    @return: d_max' value
    """
    distances = []
    for i in range(len(cluster_list)-1):
        cluster_i = cluster_list[i]
        for j in range(i+1, len(cluster_list)):
            cluster_j = cluster_list[j]
            distances.extend(fpena_get_inter_cluster_distances(
                cluster_i, cluster_j, matrix))
    return numpy.max(distances)


def fpena_get_inter_cluster_distances(cluster1, cluster2, matrix):
    distances = []
    for cluster1_element in cluster1:
        for cluster2_element in cluster2:
            distances.append(matrix[cluster1_element, cluster2_element])
    return distances


def fpena_get_intra_cluster_distances(cluster, matrix):
    distances = []
    for i in range(len(cluster)-1):
        for j in range(i+1, len(cluster)):
            distances.append(matrix[cluster[i], cluster[j]])

    if len(distances) == 0:
        # This means the cluster has only one element. We raise an exception
        # so the caller has to define its behaviour
        raise SingularClusterException("The cluster has only one element")

    return distances


def fpena_get_clusters(cluster_labels):
    clusters_map = {}
    index = 0
    for cluster_label in cluster_labels:
        if cluster_label not in clusters_map:
            clusters_map[cluster_label] = []
        clusters_map[cluster_label].append(index)
        index += 1

    return clusters_map

# my_lablels = [0, 0, 1, 0, 1, 1, 2, 0, 2]
# my_clusters = fpena_get_clusters(my_lablels)
# print(my_clusters)
#
# my_matrix = numpy.ndarray((9,9))
# print(fpena_get_intra_cluster_distances(my_clusters[0], my_matrix))
# print(fpena_get_inter_cluster_distances2(0, 1, my_clusters, my_matrix))
# print(fpena_get_inter_cluster_distances(my_clusters[0], my_clusters[1], my_matrix))
# print(fpena_max_intercluster_distance(my_clusters, my_matrix))
# print(fpena_min_intracluster_distances(my_clusters, my_matrix))
# print(fpena_evaluate(my_clusters, my_matrix))
