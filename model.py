import numpy as np
from collections import Counter


def euclidean_distance(x1: np.ndarray, x2: np.ndarray):
    return np.sqrt(np.dot(x1, x1) + np.dot(x2, x2) - 2 * np.dot(x1, x2))


def initial_centers(X, n_clusters):
    """
    Implementation of advanced (in compare with random) algorithm for choosing initial centroids
    k-means++ (http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

    Steps:
        1. Choose first point randomly
        2. Calculate for every point distance to closest center
        3. Select new center with probability proportional to this distances
        4. Repeat 2-3 until all centers selected
    """
    points_count = X.shape[0]

    if points_count <= n_clusters:
        return X

    centers = [np.random.choice(points_count)]

    for _ in range(n_clusters - 1):
        prob = np.array([min(euclidean_distance(X[i], X[c]) for c in centers) for i in range(points_count)])
        prob = prob * prob
        prob /= np.sum(prob)

        centers.append(np.random.choice(points_count, p=prob))

    return X[centers]


def label_points(X, centers):
    points_count = X.shape[0]
    labels = np.zeros(points_count, dtype=np.int)

    for idx, point in enumerate(X):
        labels[idx] = np.argmin([euclidean_distance(point, c) for c in centers])

    return labels


def find_centroids(X, labels, n_clusters):
    denominator = Counter(labels)
    n_features = X.shape[1]
    centers = np.zeros((n_clusters, n_features))

    for label, point in zip(labels, X):
        centers[label] += point / denominator[label]

    return centers


def k_means(X, n_clusters=4, max_iterations=300, convergence_threshold=1e-4, init=None):
    """
    Clustering algorithm with predefined amount of clusters (n_clusters)
    Steps:
        1. Choose initial centers or use predefined (:type init: np.array[Point]
                                                     :type Point: 1-d np.array)
        2. Calculate labels with respect to current selected centers
        3. Select new centers in labeled clusters
        4. Repeat 2,3 until centers converge or loop reach max_iterations
    """
    centers = previous_centers = initial_centers(X, n_clusters) if init is None else init

    for _ in range(max_iterations):
        labels = label_points(X, centers)
        centers = find_centroids(X, labels, n_clusters)

        # calculate shift between previous and current runs
        diff = (centers - previous_centers).reshape(-1)
        if np.dot(diff, diff) < convergence_threshold:
            break

        previous_centers = centers

    return labels.reshape(-1, 1)
