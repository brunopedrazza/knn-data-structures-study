from helpers.utils import euclidean_distance
from helpers.heap import MaxHeap
from trees.nodes.vptreenode import VpTreeNode
from trees.tree import ClassificationTree


class VpTree(ClassificationTree):

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
