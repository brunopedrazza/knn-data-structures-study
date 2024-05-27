import numpy as np

from helpers.utils import euclidean_distance
from helpers.heap import MaxHeap
from trees.nodes.node import Node


class ClassificationTree:

    def __init__(self, X, k, leaf_size, node: Node):
        self._root = node(np.array(X), np.array([i for i in range(0, len(X))]), leaf_size)
        self._k = k
        self.distance_count = 0

    def predict(self, X, prediction_method):
        best_idxs = np.empty((X.shape[0], self._k), dtype=np.int32)
        
        for i, x in enumerate(X):
            mh = prediction_method(self._root, x, MaxHeap(k=self._k))
            best_idxs[i] = np.array(mh.heap)[:, 1]

        return best_idxs
    
    def calculate_distances_leaf(self, X, X_idx, target, mh: MaxHeap):
        dists = euclidean_distance(np.array([target]), X)[0]
        self.distance_count += X.shape[0]

        sorted_idxs = np.argsort(dists)
        sorted_dists = dists[sorted_idxs[:self._k]]
        sorted_X_idxs = X_idx[sorted_idxs[:self._k]]
        if len(mh.heap) == 0 or mh.heap[0][0] > sorted_dists[0]:
            for d, idx in zip(sorted_dists, sorted_X_idxs):
                if len(mh.heap) < self._k or d < mh.heap[0][0]:
                    mh.add([d, idx])
        return mh