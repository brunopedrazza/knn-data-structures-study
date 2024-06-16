import numpy as np

from trees.nodes.node import Node


class KdTreeNode(Node):
    """ Class that defines a KdTree node. Used to construct a KdTree structure.

    Steps to construct the tree:
    1 - An axis is defined based on the depth of the node and the dimension of the points.
    2 - The points are sorted based on the value on the axis.
    3 - Split the points in half and divide them on the left and right childs.

    To use for classification time, it stores the split value and the axis.
    The split value will be used as a reference to determine whether go to the right or to the left given a target point.

    Construction time complexity is O(n log n).
    """

    def __init__(self, X, X_idx, leaf_size, depth=0):
        """ Init method to construct the tree structure.

        Parameters
        ----------
        X : List[Any]
            Construction points.
        X_idx : List[Any]
            Indices of the X points in the training set.
        leaf_size : int
            Number of points in leaves.
        depth : int
            The depth of the node in the tree.
        """

        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.split_value = None
        self.axis = None
        self.left = self.right = None
        
        n = X.shape[0]

        if n <= leaf_size:
            self.create_leaf(X, X_idx)
            return
        
        axis = depth % X.shape[1]

        idx_array = np.argsort(X[:, axis])
        X = X[idx_array]
        X_idx = X_idx[idx_array]
        
        mid = n // 2
        self.split_value = X[mid][axis]
        self.axis = axis

        if mid > 0:
            self.left = KdTreeNode(X[:mid], X_idx[:mid], leaf_size, depth + 1)
        
        if n - (mid + 1) > 0:
            self.right = KdTreeNode(X[mid:], X_idx[mid:], leaf_size, depth + 1)
