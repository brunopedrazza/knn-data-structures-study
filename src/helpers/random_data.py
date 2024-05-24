from sklearn.cluster import KMeans
import numpy as np

def generate_point_from_existing(data):
    new_point = np.array([np.random.choice(data[:, i]) for i in range(data.shape[1])]).reshape(1, -1)
    return new_point

def generate_point_statistical(data):
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    new_point = np.random.normal(means, stds).reshape(1, -1)
    return new_point

def generate_point_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    cluster_centers = kmeans.cluster_centers_
    selected_center = cluster_centers[np.random.choice(range(n_clusters))]
    
    # Generate a new point around the selected cluster center
    perturbation = np.random.normal(0, 0.1, size=selected_center.shape)
    new_point = selected_center + perturbation
    return new_point.reshape(1, -1)