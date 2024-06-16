from helpers.utils import euclidean_distance
from helpers.heap import MaxHeap
from trees.nodes.vptreenode import VpTreeNode
from trees.tree import ClassificationTree


class VpTree(ClassificationTree):
    """ Class that defines a VpTree structure. Used to classify a target point.
    
    To traverse the tree recursively, the distance is calculated between the target point and the current vantage point. If the distances is 
    less than the threshold, it goes to the closer child in relation to the vantage point, and to the farther if not. It keeps going to the 
    "good" child until it reaches a leaf node.
    After going all the way to the "good" side, it have to check if it needs to check the "bad" side as well. If the difference from the 
    threshold to the distance of the target to the vantage point is less than the distance of most distant neighbor found, 
    it needs to check for the "bad" side.

    The closest neighbors are stored in a max heap structure. It is populated when it reaches a leaf node.
    """

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, VpTreeNode)

    def __predict(self, current: VpTreeNode, target, mh: MaxHeap):
        if not current.is_leaf:
            d = euclidean_distance(target, current.vp)
            if d < current.t:
                good = current.closer
                bad = current.farther
            else:
                good = current.farther
                bad = current.closer
            
            mh = self.__predict(good, target, mh)
            
            r_ = abs(current.t - d)
            if not mh.is_full() or mh.heap[0][0] > r_:
                mh = self.__predict(bad, target, mh)
                
            return mh
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh)
         
    
    def predict(self, X):
        return super().predict(X, self.__predict)
