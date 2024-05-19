import numpy as np

from heap import MaxHeap
from utils import euclidean_distance
    
class KdTreeNode:

    def __init__(self, X, X_idx, leaf_size, depth=0):

        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.split_value = None
        self.axis = None
        self.left = self.right = None
        
        axis = depth % X.shape[1]

        idx_array = np.argsort(X[:, axis])
        X = X[idx_array]
        X_idx = X_idx[idx_array]

        n = X.shape[0]

        if n <= leaf_size:
            self.X = X
            self.X_idx = X_idx
            self.is_leaf = True
            return
        
        mid = n // 2
        self.split_value = X[mid][axis]
        self.axis = axis

        if mid > 0:
            self.left = KdTreeNode(X[:mid], X_idx[:mid], leaf_size, depth + 1)
        
        if n - (mid + 1) > 0:
            self.right = KdTreeNode(X[mid:], X_idx[mid:], leaf_size, depth + 1)

class KdTree:

    def __init__(self, X, k, leaf_size):
        self._root = KdTreeNode(np.array(X), np.array([i for i in range(0, len(X))]), leaf_size)
        self._k = k
        self.distance_count = 0

    def __predict(self, current: KdTreeNode, target, mh: MaxHeap):

        if not current.is_leaf:
            axis = current.axis
            if target[axis] < current.split_value:
                good, bad = current.left, current.right
            else:
                good, bad = current.right, current.left
            
            mh = self.__predict(good, target, mh)

            r_ = target[axis] - current.split_value
            if not mh.is_full() or mh.heap[0][0] >= abs(r_):
                mh = self.__predict(bad, target, mh)
        else:
            dists = euclidean_distance(np.array([target]), current.X)[0]
            self.distance_count += current.X.shape[0]

            sorted_idxs = np.argsort(dists)
            sorted_dists = dists[sorted_idxs[:self._k]]
            sorted_X_idxs = current.X_idx[sorted_idxs[:self._k]]
            if len(mh.heap) == 0 or mh.heap[0][0] > sorted_dists[0]:
                for d, idx in zip(sorted_dists, sorted_X_idxs):
                    if len(mh.heap) < self._k or d < mh.heap[0][0]:
                        mh.add([d, idx])
            return mh

        return mh
    
    def predict(self, X, k):
        best_idxs = np.empty((X.shape[0], k), dtype=np.int32)
        
        for i, x in enumerate(X):
            mh = self.__predict(self._root, x, MaxHeap(k=k))
            best_idxs[i] = np.array(mh.heap)[:, 1]

        return best_idxs
