import numpy as np

from helpers.utils import euclidean_distance
from helpers.heap import MaxHeap
from trees.nodes.node import Node


class ClassificationTree:

    def __init__(self, X, k, leaf_size, node: Node):
        self._root = node(X=np.array(X), X_idx=np.arange(len(X)), leaf_size=leaf_size)
        self._k = k
        self.distance_count = 0

    def predict(self, X, prediction_method):
        """ Method that iterates through the testing points and use the prediction 
        method to get the k close neighbors. Returns their indices on the training set.

        Parameters
        ----------
        X : List[Any]
            Points to classify.
        prediction_method : callable
            The tree method to use for predicting.
        """

        best_idxs = np.empty((X.shape[0], self._k), dtype=np.int32)
        
        for i, x in enumerate(X):
            mh = prediction_method(self._root, x, MaxHeap(k=self._k))
            best_idxs[i] = np.array(mh.heap)[:, 1]

        return best_idxs
    
    
    def calculate_distances_leaf(self, X, X_idx, target, mh: MaxHeap):
        """ Method that calculates the distances of the points on the leaf node 
        to the the target point and stores the closer ones into the max heap.

        Parameters
        ----------
        X : List[Any]
            Points of the leaf node.
        X_idx : List[Any]
            Indices of the X points in the training set.
        target : Any
            The target point.
        mh : MaxHeap
            The max heap to store best distances.
        """

        dists = euclidean_distance(np.array([target]), X)[0]
        self.distance_count += X.shape[0]

        sorted_idxs = np.argsort(dists)
        sorted_dists = dists[sorted_idxs[:self._k]]
        sorted_X_idxs = X_idx[sorted_idxs[:self._k]]

        index = 0
        while index < len(sorted_dists) and (len(mh.heap) < self._k or sorted_dists[index] < mh.heap[0][0]):
            mh.add([sorted_dists[index], sorted_X_idxs[index]])
            index += 1
        return mh
