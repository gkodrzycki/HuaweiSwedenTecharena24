from sklearn.cluster import KMeans
import numpy as np


def create_model(n_clusters=256):
    return KMeans(n_clusters=n_clusters, n_jobs=-1)


def train_model(model, X):
    model.fit(X)


def get_centroids(model):
    return model.cluster_centers_


def get_deep_features(model, X):
    centroids = get_centroids(model)
    distances = np.array([np.sqrt(np.sum((X - centroid) ** 2, axis=1)) for centroid in centroids])
    return distances.T