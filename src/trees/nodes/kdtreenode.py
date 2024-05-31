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


class KdTreeOptNode(Node):

    def __init__(self, X, X_idx, leaf_size):

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
        
        # calculate variance along each dimension
        variances = np.var(X, axis=0)
        
        # select the axis with the maximum variance
        axis = np.argmax(variances)

        idx_array = np.argsort(X[:, axis])
        X = X[idx_array]
        X_idx = X_idx[idx_array]

        mid = n // 2
        self.split_value = X[mid][axis]
        self.axis = axis

        if mid > 0:
            self.left = KdTreeOptNode(X[:mid], X_idx[:mid], leaf_size)
        if n - (mid + 1) > 0:
            self.right = KdTreeOptNode(X[mid:], X_idx[mid:], leaf_size)