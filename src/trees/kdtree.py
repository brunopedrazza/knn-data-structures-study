from helpers.heap import MaxHeap

from trees.nodes.kdtreenode import KdTreeNode
from trees.tree import ClassificationTree


class KdTree(ClassificationTree):
    """ Class that defines a KdTree structure. Used to classify a target point.
    
    To traverse the tree recursively, it checks if the value on axis of the target point is less than the split value, it goes to 
    the left if it's true or to the right if not. It keeps going to the "good" child until it reaches a leaf node.
    After going all the way to the "good" side, it have to check if it needs to check the "bad" side as well. If the difference 
    from the value of the target on axis to the split value is less than the distance of most distant neighbor found, it needs 
    to check for the "bad" side.

    The closest neighbors are stored in a max heap structure. It is populated when it reaches a leaf node.
    """

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, KdTreeNode)

    def __query(self, current: KdTreeNode, target, mh: MaxHeap, depth=0):
        if not current.is_leaf:
            axis = current.axis
            if target[axis] < current.split_value:
                good = current.left
                bad = current.right
            else:
                good = current.right
                bad = current.left
            
            mh = self.__query(good, target, mh, depth+1)

            r_ = abs(target[axis] - current.split_value)
            if not mh.is_full() or mh.heap[0][0] > r_:
                mh = self.__query(bad, target, mh, depth+1)
                
            return mh
        else:
            return super().calculate_distances_leaf(current.X, current.X_idx, target, mh, depth)

    
    def predict(self, X):
        return super().query(X, self.__query)
