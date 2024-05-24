from trees.nodes.kd_tree_node import KdTreeNode
from helpers.heap import MaxHeap
from trees.tree import ClassificationTree


class KdTree(ClassificationTree):

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, KdTreeNode)

    def __predict(self, current: KdTreeNode, target, mh: MaxHeap):
        if not current.is_leaf:
            axis = current.axis
            if target[axis] < current.split_value:
                good, bad = current.left, current.right
            else:
                good, bad = current.right, current.left
            
            mh = self.__predict(good, target, mh)

            r_ = target[axis] - current.split_value
            if not mh.is_full() or mh.heap[0][0] >= abs(r_):
                mh = self.__predict(bad, target, mh)
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh)

        return mh
    
    def predict(self, X, k):
        return super().predict(X, k, self.__predict)
