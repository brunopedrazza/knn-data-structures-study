from trees.nodes.kdtreenode import KdTreeNode
from helpers.heap import MaxHeap
from trees.tree import ClassificationTree


class KdTree(ClassificationTree):

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, KdTreeNode)

    def __predict(self, current: KdTreeNode, target, mh: MaxHeap):
        if not current.is_leaf:
            axis = current.axis
            if target[axis] < current.split_value:
                good = current.left
                bad = current.right
            else:
                good = current.right
                bad = current.left
            
            mh = self.__predict(good, target, mh)

            r_ = abs(target[axis] - current.split_value)
            if not mh.is_full() or mh.heap[0][0] >= r_:
                mh = self.__predict(bad, target, mh)
                
            return mh
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh)

    
    def predict(self, X):
        return super().predict(X, self.__predict)
