
import math
from helpers.utils import euclidean_distance
from helpers.heap import MaxHeap
from trees.nodes.vptreenode import VpTreeNode, VpTreeNode2
from trees.tree import ClassificationTree


class VpTree(ClassificationTree):

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, VpTreeNode)

    def __predict(self, current: VpTreeNode, target, mh: MaxHeap):
        if not current.is_leaf:
            d = euclidean_distance(target, current.vp)
            if d < current.t:
                good, bad = current.closer, current.farther
            else:
                good, bad = current.farther, current.closer
            
            mh = self.__predict(good, target, mh)
            
            r_ = abs(current.t - d)
            if not mh.is_full() or mh.heap[0][0] > r_:
                mh = self.__predict(bad, target, mh)
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh)
         
        return mh
    
    def predict(self, X):
        return super().predict(X, self.__predict)

class VpTree2(ClassificationTree):

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, VpTreeNode2)

    def __predict(self, current: VpTreeNode2, target, mh: MaxHeap):
        if not current.is_leaf:
            d = euclidean_distance(target, current.vp)
            good, bad = None, None
            if not mh.is_full():
                if d < current.closer_max:
                    good = current.closer
                    bad = current.farther
                else:
                    good = current.farther
                    bad = current.closer
            else:
                if current.closer_min - mh.heap[0][0] < d < current.farther_max + mh.heap[0][0]:
                    good = current.closer
                    bad = current.farther
                elif current.farther_min - mh.heap[0][0] <= d <= current.farther_max + mh.heap[0][0]:
                    good = current.farther
                    bad = current.closer
            
            if good:
                mh = self.__predict(good, target, mh)

            if bad and mh.heap[0][0] > abs(current.farther_min - d):
                mh = self.__predict(bad, target, mh)

            return mh
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh)
        
    
    def predict(self, X):
        return super().predict(X, self.__predict)