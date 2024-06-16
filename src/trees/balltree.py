import numpy as np

from helpers.heap import MaxHeap
from trees.nodes.balltreenode import BallTreeNode
from trees.tree import ClassificationTree


class BallTree(ClassificationTree):

    """ Class that defines a BallTree structure. Used to classify a target point.
    
    To traverse the tree recursively, the target point is projected on the line vector and it goes to the left if the projection
    is less than or equal to the current median or to the right if not. It keeps going to the "good" child until it reaches a leaf node.
    After going all the way to the "good" side, it have to check if it needs to check the "bad" side as well. If the distance 
    to the hyperplane is less than the distance of most distant neighbor found, it needs to check for the "bad" side.

    The closest neighbors are stored in a max heap structure. It is populated when it reaches a leaf node.
    """

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, BallTreeNode)

    def __predict(self, current: BallTreeNode, target, mh: MaxHeap):
        if not current.is_leaf:
            projection = np.dot(target, current.line_vector) / current.vector_norm
            if projection <= current.median:
                good = current.left
                bad = current.right
            else:
                good = current.right
                bad = current.left
            
            mh = self.__predict(good, target, mh)

            distance_to_hyperplane = abs(projection - current.median)
            if not mh.is_full() or mh.heap[0][0] > distance_to_hyperplane:
                mh = self.__predict(bad, target, mh)
            
            return mh
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh)
        
    
    def predict(self, X):
        return super().predict(X, self.__predict)
