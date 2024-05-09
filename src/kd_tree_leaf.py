import numpy as np

from heap import MaxHeap
from utils import euclidean_distance
    
class KdTreeLeaf:

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
            self.left = KdTreeLeaf(X[:mid], X_idx[:mid], leaf_size, depth + 1)
        
        if n - (mid + 1) > 0:
            self.right = KdTreeLeaf(X[mid:], X_idx[mid:], leaf_size, depth + 1)

    
    @staticmethod
    def construct(X, leaf_size):
        if not hasattr(X, "dtype"):
            X = np.array(X)
        return KdTreeLeaf(X, np.array([i for i in range(0, X.shape[0])]), leaf_size=leaf_size)

    @staticmethod
    def __predict(current, target, best_idx=None, best_d=None):

        if not current.is_leaf:
            axis = current.axis
            if target[axis] < current.split_value:
                good, bad = current.left, current.right
            else:
                good, bad = current.right, current.left
            
            best_idx_leaf, best_d_leaf = KdTreeLeaf.__predict(good, target, best_idx, best_d)
            if best_idx is None or best_d_leaf < best_d:
                best_idx = best_idx_leaf
                best_d = best_d_leaf

            r_ = target[axis] - current.split_value
            if best_d >= abs(r_):
                best_idx_leaf, best_d_leaf = KdTreeLeaf.__predict(bad, target, best_idx, best_d)
                if best_d_leaf < best_d:
                    best_idx = best_idx_leaf
                    best_d = best_d_leaf
        else:
            dists = euclidean_distance(np.array([target]), current.X)
            best_idx_leaf = np.argsort(dists, axis=1)[:, :1][0][0]
            return current.X_idx[best_idx_leaf], dists[0][best_idx_leaf]

        return best_idx, best_d
    
    def predict(self, X, k):
        best_idxs = np.empty((X.shape[0], k), dtype=np.int32)
        for i, x in enumerate(X):
            xp = self.__predict(self, x)
            best_idxs[i] = xp[0]
        return best_idxs

