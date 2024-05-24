import numpy as np

from trees.nodes.ball_tree_node import BallTreeNode
from helpers.heap import MaxHeap
from trees.tree import ClassificationTree


class BallTree(ClassificationTree):

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, BallTreeNode)

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
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh)
        
        return mh
    
    def predict(self, X, k):
        return super().predict(X, k, self.__predict)
