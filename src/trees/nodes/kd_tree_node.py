import numpy as np

from trees.nodes.node import Node


class KdTreeNode(Node):

    def __init__(self, X, X_idx, leaf_size, depth=0):

        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.split_value = None
        self.axis = None
        self.left = self.right = None
        
        axis = depth % X.shape[1]

        idx_array = np.argsort(X[:, axis])
        X = X[idx_array]
        X_idx = X_idx[idx_array]

        n = X.shape[0]

        if n <= leaf_size:
            self.create_leaf(X, X_idx)
            return
        
        mid = n // 2
        self.split_value = X[mid][axis]
        self.axis = axis

        if mid > 0:
            self.left = KdTreeNode(X[:mid], X_idx[:mid], leaf_size, depth + 1)
        
        if n - (mid + 1) > 0:
            self.right = KdTreeNode(X[mid:], X_idx[mid:], leaf_size, depth + 1)