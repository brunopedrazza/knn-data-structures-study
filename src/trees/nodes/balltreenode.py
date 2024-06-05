import numpy as np

from helpers.utils import find_farthest_point
from trees.nodes.node import Node


class BallTreeNode(Node):

    def __init__(self, X, X_idx, leaf_size):
        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.left = self.right = None
        self.median = None
        self.transp = None
        
        n = X.shape[0]
        if n <= leaf_size:
            self.create_leaf(X, X_idx)
            return
        
        rand_idx = np.random.randint(n)
        rand_point = X[rand_idx]
        
        # Find the most distant point from the randomly picked point
        x0 = find_farthest_point(X, rand_point)
        
        # Find the point most distant from the farthest point
        x1 = find_farthest_point(X, x0)

        # Calculate the line vector between them
        self.line_vector = x1 - x0
        
        # Project all points onto this line
        self.transp = np.linalg.norm(self.line_vector)
        projections = np.dot(X, self.line_vector) / self.transp
        
        # Split based on the median of the projections
        self.median = np.median(projections)
        left_idx = projections <= self.median
        right_idx = projections > self.median
        
        self.left = BallTreeNode(X[left_idx], X_idx[left_idx], leaf_size)
        self.right = BallTreeNode(X[right_idx], X_idx[right_idx], leaf_size)
