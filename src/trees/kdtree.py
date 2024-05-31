from trees.nodes.kdtreenode import KdTreeNode, KdTreeOptNode, Node
from helpers.heap import MaxHeap
from trees.tree import ClassificationTree


class KdTree(ClassificationTree):

    def __init__(self, X, k, leaf_size, optimized=False):
        node = KdTreeOptNode if optimized else KdTreeNode
        super().__init__(X, k, leaf_size, node)

    def __predict(self, current: Node, target, mh: MaxHeap):
        if not current.is_leaf:
            axis = current.axis
            if target[axis] < current.split_value:
                good, bad = current.left, current.right
            else:
                good, bad = current.right, current.left
            
            mh = self.__predict(good, target, mh)

            r_ = abs(target[axis] - current.split_value)
            if not mh.is_full() or mh.heap[0][0] >= r_:
                mh = self.__predict(bad, target, mh)
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh)

        return mh
    
    def predict(self, X):
        return super().predict(X, self.__predict)
