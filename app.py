import numpy as np
from flask import Flask, request, abort

from model import k_means

app = Flask(__name__)


@app.route('/clustering/labels', methods=['POST'])
def clustering():
    try:
        n_clusters = int(request.args.get('num_clusters', 4))
        assert n_clusters > 0
    except (ValueError, AssertionError):
        raise abort(400, 'num_clusters must be a positive integer')

    try:
        max_iterations = int(request.args.get('max_iterations', 300))
        assert max_iterations > 0
    except (ValueError, AssertionError):
        raise abort(400, 'max_iterations must be a positive integer')

    try:
        points = parse_input(request.data.decode('ascii'))
    except ValueError:
        raise abort(400, 'Input data must be matrix MxN of float numbers')

    if not points.size:
        raise abort(400, 'Input matrix cannot be empty')

    labels = k_means(points, n_clusters, max_iterations)
    labels += 1  # normalize labels to begin from 1

    return '[{}]'.format(';\n'.join(
        ', '.join(f'{x}' for x in list(row) + list(label))
        for row, label in zip(points, labels))
    )


def parse_input(data: str):
    """
    :param data: input string with matrix MxN, rows splitted by ; cols by ,
            [51.1, 30.2;
             64.91, 51.67;
             70.45, 68.7;
             61.9, 45.2]

    :return: np.ndarray with shape MxN
    """
    rows = data.count(';') + 1
    data = data.replace(';', ',').strip('[]')
    return np.fromstring(data, sep=',').reshape(rows, -1)
