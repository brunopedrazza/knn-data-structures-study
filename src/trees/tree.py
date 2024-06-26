import numpy as np

from helpers.heap import MaxHeap
from trees.nodes.node import Node


class ClassificationTree:

    def __init__(self, X, k, leaf_size, node: Node):
        self._root = node(X=np.array(X), X_idx=np.arange(len(X)), leaf_size=leaf_size)
        self._k = k
        self.total_points_visited = 0
        self.max_depth = 0

    def query(self, X, prediction_method):
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
    
    
    def calculate_distances_leaf(self, X, X_idx, target, mh: MaxHeap, depth):
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
        depth : int
            The depth of the tree.
        """

        dists = np.linalg.norm(X - target, axis=1)
        self.total_points_visited += X.shape[0]
        self.max_depth = max(self.max_depth, depth)

        sorted_idxs = np.argsort(dists)
        sorted_dists = dists[sorted_idxs[:self._k]]
        sorted_X_idxs = X_idx[sorted_idxs[:self._k]]

        index = 0
        while index < len(sorted_dists) and (len(mh.heap) < self._k or sorted_dists[index] < mh.heap[0][0]):
            mh.add([sorted_dists[index], sorted_X_idxs[index]])
            index += 1
        return mh
