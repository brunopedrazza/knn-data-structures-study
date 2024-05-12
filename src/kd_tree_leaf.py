import numpy as np
import heapq

from heap import MaxHeap
from utils import euclidean_distance
    
class KdTreeLeafNode:

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
            self.left = KdTreeLeafNode(X[:mid], X_idx[:mid], leaf_size, depth + 1)
        
        if n - (mid + 1) > 0:
            self.right = KdTreeLeafNode(X[mid:], X_idx[mid:], leaf_size, depth + 1)

class KdTreeLeaf:

    def __init__(self, X, k, leaf_size):
        self._root = KdTreeLeafNode(X, np.array([i for i in range(0, len(X))]), leaf_size)
        self._k = k
        self.distance_count = 0

    def __predict(self, current: KdTreeLeafNode, target, mh: MaxHeap):

        if not current.is_leaf:
            axis = current.axis
            if target[axis] < current.split_value:
                good, bad = current.left, current.right
            else:
                good, bad = current.right, current.left
            
            mh = self.__predict(good, target, mh)

            r_ = target[axis] - current.split_value
            if mh.heap[0][0] >= abs(r_) or not mh.is_full():
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
    
    def __predict2(self, current: KdTreeLeafNode, target, mh):

        if not current.is_leaf:
            axis = current.axis
            if target[axis] < current.split_value:
                good, bad = current.left, current.right
            else:
                good, bad = current.right, current.left
            
            mh = self.__predict2(good, target, mh)

            r_ = target[axis] - current.split_value
            if len(mh) < self._k or -mh[0][0] > abs(r_):  # Use -mh[0][0] to get the actual max distance
                mh = self.__predict2(bad, target, mh)
        else:
            dists = euclidean_distance(np.array([target]), current.X)[0]
            self.distance_count += current.X.shape[0]

            for d, idx in zip(dists, current.X_idx):
                if len(mh) < self._k:
                    heapq.heappush(mh, (-d, idx))  # Push negative distance to simulate max heap
                elif -mh[0][0] > d:  # Check if the largest (negative) distance is greater than the current distance
                    heapq.heapreplace(mh, (-d, idx))  # Replace the root and then heapify
            return mh

        return mh
    
    def predict(self, X, k):
        best_idxs = np.empty((X.shape[0], k, 2), dtype=np.int32)
        for i, x in enumerate(X):
            mh = self.__predict(self._root, x, MaxHeap(k=k))
            best_idxs[i] = mh.heap

        indices = np.argsort(best_idxs[:,:,0], axis=1)
        return best_idxs[np.arange(best_idxs.shape[0])[:, None], indices, 1]

    def predict2(self, X, k):
        best_idxs = np.empty((X.shape[0], k, 2), dtype=np.int32)
        for i, x in enumerate(X):
            best_idxs[i] = self.__predict2(self._root, x, [])

        indices = np.argsort(best_idxs[:,:,0], axis=1)
        return best_idxs[np.arange(best_idxs.shape[0])[:, None], indices, 1]

