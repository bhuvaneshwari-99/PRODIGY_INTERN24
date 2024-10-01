import random

import numpy as np

# Step 1: Create a mock dataset
# Let's simulate a small dataset with customer purchase history (total spent, frequency of purchases)
data = np.array([
    [500, 10], [1500, 35], [250, 7], [1200, 30], [350, 12],
    [2750, 50], [300, 8], [3500, 70], [600, 18], [800, 25],
    [1300, 30], [900, 20], [2100, 40], [3200, 65], [750, 25],
    [1600, 37], [450, 9], [2000, 48], [2700, 55], [1250, 30]
])

# Step 2: Define the K-means clustering algorithm
def kmeans(data, k, max_iters=100):
    # Randomly initialize k centroids from the data
    centroids = data[random.sample(range(len(data)), k)]

    for _ in range(max_iters):
        # Step 3: Assign points to the nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            # Compute distances to each centroid
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            # Assign the point to the nearest cluster (centroid)
            nearest_centroid = np.argmin(distances)
            clusters[nearest_centroid].append(point)

        # Step 4: Update centroids (calculate the mean of the points in each cluster)
        new_centroids = [np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)]

        # Step 5: Check for convergence (if centroids don't change, break the loop)
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids

# Step 6: Run the K-means algorithm with k = 3 (3 customer groups)
k = 3
clusters, centroids = kmeans(data, k)

# Step 7: Display the results
print(f"Clusters (k={k}):")
for idx, cluster in enumerate(clusters):
    print(f"Cluster {idx + 1}:")
    for point in cluster:
        print(f"Customer Purchase History: Total Spent = {point[0]}, Frequency = {point[1]}")
    print()

# Display the centroids
print("Centroids:")
for idx, centroid in enumerate(centroids):
    print(f"Centroid {idx + 1}: Total Spent = {centroid[0]:.2f}, Frequency = {centroid[1]:.2f}")
