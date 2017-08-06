from collections import Counter

import pandas
import numpy as np
from sklearn.cluster import KMeans

from model import k_means, label_points, find_centroids, initial_centers
from app import app


DATASET_URL = 'https://raw.githubusercontent.com/datascienceinc/learn-data-science/master/' \
              'Introduction-to-K-means-Clustering/Data/data_1024.csv'


def test_model():
    """ Comparing results between our implementation and SKLearn with same set of init centers"""

    df = pandas.read_csv(DATASET_URL, sep='\t')
    X = df[['Distance_Feature', 'Speeding_Feature']].as_matrix()
    np.random.shuffle(X)
    X = X[:500]

    test_centers = initial_centers(X, 4)

    our_result = k_means(X, n_clusters=4, init=test_centers, convergence_threshold=1e-6)
    sample_result = KMeans(n_clusters=4, init=test_centers, n_init=1, algorithm='full').fit(X)

    our_pmf = Counter(our_result.reshape(-1)).values()
    sample_pmf = Counter(sample_result.labels_.reshape(-1)).values()

    print(our_pmf, sample_pmf)
    return set(our_pmf) == set(sample_pmf)


def test_labeling():
    p = np.array([[-2, 5], [10, 3],
                  [0, 0], [8, 8],
                  [2, -2], [5, 10]])
    centers = np.array([[1, 1], [7, 7]])
    labels = label_points(p, centers)

    assert list(labels) == [0, 1, 0, 1, 0, 1]


def test_centroids():
    p = np.array([[-2, 5], [8, 3],
                  [0, 0], [8, 8],
                  [2, -2], [5, 10]])
    centroids = find_centroids(p, [0, 1, 0, 1, 0, 1], 2)

    assert list(centroids[0]) == [0, 1]
    assert list(centroids[1]) == [7, 7]


def test_input_validation():
    with app.test_client() as c:
        resp = c.post('/clustering/labels?num_clusters=-1')

    assert resp.status_code == 400
    assert b'num_clusters must be a positive integer' in resp.get_data()

    with app.test_client() as c:
        resp = c.post('/clustering/labels?num_clusters=1&max_iterations=a')

    assert resp.status_code == 400
    assert b'max_iterations must be a positive integer' in resp.get_data()


def test_matrix_validation():
    with app.test_client() as c:
        resp = c.post('/clustering/labels', data='[0, 0; 0]')

    assert resp.status_code == 400
    assert b'Input data must be matrix MxN of float numbers' in resp.get_data()

    with app.test_client() as c:
        resp = c.post('/clustering/labels', data='[0, 0; 0 NaN]')

    assert resp.status_code == 400
    assert b'Input data must be matrix MxN of float numbers' in resp.get_data()

    with app.test_client() as c:
        resp = c.post('/clustering/labels', data='[]')

    assert resp.status_code == 400
    assert b'Input matrix cannot be empty' in resp.get_data()


def test_success():
    with app.test_client() as c:
        resp = c.post('/clustering/labels?num_clusters=2', data='[0, 0; 1, 2; 5, 5]')

    assert resp.status_code == 200
    d = resp.get_data()

    assert d in (b'[0.0, 0.0, 1;\n1.0, 2.0, 1;\n5.0, 5.0, 2]',
                 b'[0.0, 0.0, 2;\n1.0, 2.0, 2;\n5.0, 5.0, 1]')
