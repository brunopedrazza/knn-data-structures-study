import numpy as np

from heap import MaxHeap
from utils import euclidean_distance
    
class KdTreeLeaf:

    def __init__(self, X, X_idx, leaf_size, depth=0):

        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.split_value = None
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
    def __predict(current, target, best=None, best_d=None, depth=0):
        if not current:
            return best, best_d
        
        d = euclidean_distance(current.point, target)
        if d < best_d:
            best_d = d
            best = current.point

        axis = depth % len(current.point)
        if target[axis] < current.point[axis]:
            good, bad = current.left, current.right
        else:
            good, bad = current.right, current.left
        
        best, best_d = KdTreeLeaf.__predict(good, target, best, best_d, depth+1)
        r_ = target[axis] - current.point[axis]

        if best_d >= r_:
            best, best_d = KdTreeLeaf.__predict(bad, target, best, best_d, depth+1)

        return best, best_d
    
    # def predict(self, X, k):
    #     best_idxs = np.empty((X.shape[0], k, 2), dtype=np.int32)
    #     for i, x in enumerate(X):
    #         mh = self.__predict(self, x, MaxHeap(k=k))
    #         # print(mh.count)
    #         best_idxs[i] = mh.heap

    #     indices = np.argsort(best_idxs[:,:,0], axis=1)
    #     return best_idxs[np.arange(best_idxs.shape[0])[:, None], indices, 1]

