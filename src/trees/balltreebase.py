import numpy as np
\
from helpers.heap import MaxHeap
from trees.nodes.balltreenodebase import BallTreeNodeBase

from trees.tree import ClassificationTree

class BallTreeBase(ClassificationTree):

    """ Class that defines a BallWTree structure. Used to classify a target point.
    
    To traverse the tree recursively, we check for the proximity of the query point to the left and the right centroid.
    If the point is closer to the left centroid, we go to the left child, and to the right if not.
    After going all the way to the "good" side, it have to check if it needs to check the "bad" side as well. If the distance 
    to the hypersphere is less than the distance of most distant neighbor found, it needs to check for the "bad" side.

    The closest neighbors are stored in a max heap structure. It is populated when it reaches a leaf node.
    """
    
    def __query(self, current: BallTreeNodeBase, target, mh: MaxHeap, depth=0):
        if not current.is_leaf:
            dist_left = np.linalg.norm(target - current.left.centroid)
            dist_right = np.linalg.norm(target - current.right.centroid)
            if dist_left < dist_right:
                good = current.left
                bad = current.right
                bad_dist = dist_right
            else:
                good = current.right
                bad = current.left
                bad_dist = dist_left
            
            mh = self.__query(good, target, mh, depth+1)

            distance_to_hypersphere = abs(bad_dist - bad.radius)
            if not mh.is_full() or mh.heap[0][0] > distance_to_hypersphere:
                mh = self.__query(bad, target, mh, depth+1)
            
            return mh
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh, depth)
        
    
    def predict(self, X):
        return super().query(X, self.__query)