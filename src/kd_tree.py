import numpy as np

from heap import MaxHeap
from utils import euclidean_distance



class KdTreeNode:

    def __init__(self, X, X_idx, depth=0):
        if not hasattr(X, "dtype"):
            X = np.array(X)
        if not hasattr(X_idx, "dtype"):
            X_idx = np.array(X_idx)
        
        if X.shape[0] != len(X_idx):
            raise ValueError("X must have the same size of y")
        
        axis = depth % X.shape[1]

        idx_array = np.argsort(X[:, axis])
        X = X[idx_array]
        X_idx = X_idx[idx_array]

        n = X.shape[0]
        mid = n // 2

        self.point = X[mid]
        self.idx = X_idx[mid]
        self.left = self.right = self.parent = None

        if mid > 0:
            self.left = KdTreeNode(X[:mid], X_idx[:mid], depth + 1)
            self.left.parent = self
        
        if n - (mid + 1) > 0:
            self.right = KdTreeNode(X[mid+1:], X_idx[mid+1:], depth + 1)
            self.right.parent = self


class KdTree:

    def __init__(self, X):
        self._root = KdTreeNode(X, [i for i in range(0, len(X))])
        self.distance_count = 0
    
    def __predict(self, current: KdTreeNode, target, mh: MaxHeap, depth=0):
        if not current:
            return mh
        
        # get the distance squared to save 1 unecessary calculation
        # all distance are going to be compared squared
        d_squared = euclidean_distance(current.point, target, squared=True)
        self.distance_count += 1
        
        mh.add([d_squared, current.idx])

        axis = depth % len(current.point)
        if target[axis] < current.point[axis]:
            good, bad = current.left, current.right
        else:
            good, bad = current.right, current.left
        
        mh = self.__predict(good, target, mh, depth+1)
        r_ = target[axis] - current.point[axis]

        # if the most distant point to the target is further than the current point on axis,
        # we need to check the bad side

        # we are comparing with the square of r_ because our distances are squared
        if mh.heap[0][0] >= r_ * r_ or not mh.is_full():
            mh = self.__predict(bad, target, mh, depth+1)

        return mh
    
    def predict(self, X, k):
        best_idxs = np.empty((X.shape[0], k, 2), dtype=np.int32)
        for i, x in enumerate(X):
            mh = self.__predict(self._root, x, MaxHeap(k=k))
            best_idxs[i] = mh.heap

        indices = np.argsort(best_idxs[:,:,0], axis=1)
        return best_idxs[np.arange(best_idxs.shape[0])[:, None], indices, 1]
