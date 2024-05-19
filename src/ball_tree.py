import numpy as np

from heap import MaxHeap
from utils import euclidean_distance
    
class BallTreeNode:

    def __init__(self, X, X_idx, leaf_size, depth=0):
        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.left = self.right = None
        self.median = None
        
        n = X.shape[0]

        if n <= leaf_size:
            self.X = X
            self.X_idx = X_idx
            self.is_leaf = True
            return
        
        rand_idx = np.random.randint(X.shape[0])
        rand_point = X[rand_idx]
        
        # Find the most distant point from the randomly picked point
        farthest_point = BallTreeNode.find_most_distant_point(X, rand_point)
        
        # Find the point most distant from the farthest point
        second_farthest_point = BallTreeNode.find_most_distant_point(X, farthest_point)

        # Calculate the line vector between them
        self.line_vector = second_farthest_point - farthest_point

        # Project all points onto this line
        projections = np.dot(X, self.line_vector) / np.linalg.norm(self.line_vector)
        
        # Split based on the median of the projections
        self.median = np.median(projections)
        left_idx = projections <= self.median
        right_idx = projections > self.median

        # Recursive construction of the tree
        self.left = BallTreeNode(X[left_idx], X_idx[left_idx], leaf_size, depth+1)
        self.right = BallTreeNode(X[right_idx], X_idx[right_idx], leaf_size, depth+1)
    
    @staticmethod
    def find_most_distant_point(X, point):
        distances = np.linalg.norm(X - point, axis=1)
        farthest_idx = np.argmax(distances)
        return X[farthest_idx]


class BallTree:

    def __init__(self, X, k, leaf_size):
        self._root = BallTreeNode(np.array(X), np.array([i for i in range(0, len(X))]), leaf_size)
        self._k = k
        self.distance_count = 0

    def __predict(self, current: BallTreeNode, target, mh: MaxHeap):
        if not current.is_leaf:

            projection = np.dot(target, current.line_vector) / np.linalg.norm(current.line_vector)
            if projection <= current.median:
                good, bad = current.left, current.right
            else:
                good, bad = current.right, current.left
            
            mh = self.__predict(good, target, mh)

            distance_to_hyperplane = abs(projection - current.median)
            if mh.heap[0][0] > distance_to_hyperplane or not mh.is_full():
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
