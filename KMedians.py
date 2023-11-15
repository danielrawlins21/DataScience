import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from pyclustering.cluster.kmedoids import kmedoids

"""def kmeans_l1(data, k, initial_centers):
    n, m = data.shape
    centroids = initial_centers.copy()
    prev_centroids = np.zeros_like(centroids)

    while not np.array_equal(centroids, prev_centroids):
        # Assign each point to the nearest centroid based on L1 norm
        labels = np.argmin(np.sum(np.abs(data[:, np.newaxis] - centroids), axis=2), axis=1)

        # Update centroids using medians
        for i in range(k):
            mask = labels == i
            if np.sum(mask) > 0:
                centroids[i] = np.median(data[mask], axis=0)

        prev_centroids = centroids.copy()

    return centroids, labels

# Given dataset
data = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])

# Number of clusters
k = 2

# Initial cluster centers for k-means
initial_centers_kmeans = np.array([[-5, 2], [0, -6]])

# K-means clustering with L1 norm
kmeans_cluster_centers, kmeans_labels = kmeans_l1(data, k, initial_centers_kmeans)

# Print k-means with L1 norm results
print("K-Means (L1 Norm) Cluster Centers:")
print(kmeans_cluster_centers)
print("K-Means (L1 Norm) Labels:")
print(kmeans_labels)"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from pyclustering.cluster.kmedoids import kmedoids

# Given dataset
data = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])

# Number of clusters
k = 2

# Initial cluster centers for k-means
initial_centers_kmeans = np.array([[-5, 2], [0, -6]])

# K-means clustering
kmeans = KMeans(n_clusters=k, init=initial_centers_kmeans, n_init=1, random_state=42)
kmeans.fit(data)
kmeans_cluster_centers = kmeans.cluster_centers_
kmeans_labels = kmeans.labels_

# Print k-means results
print("K-Means Cluster Centers:")
print(kmeans_cluster_centers)
print("K-Means Labels:")
print(kmeans_labels)

# Initial medoids for k-medoids
initial_medoids = [1, 0]  # Indices corresponding to [(-5, 2), (0, -6)]

# K-medoids clustering
kmedoids_instance = kmedoids(data, initial_medoids)
kmedoids_instance.process()
kmedoids_cluster_centers = data[kmedoids_instance.get_medoids()]
kmedoids_labels, _ = pairwise_distances_argmin_min(data, kmedoids_cluster_centers)

# Print k-medoids results
print("\nK-Medoids Cluster Centers:")
print(kmedoids_cluster_centers)
print("K-Medoids Labels:")
print(kmedoids_labels)
