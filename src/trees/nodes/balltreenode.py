import numpy as np

from helpers.utils import find_farthest_point
from trees.nodes.node import Node


class BallTreeNode(Node):
    """ Class that defines a BallTree node. Used to construct a BallTree structure.

    Steps to construct the tree:
    1 - Pick a random point in X.
    2 - Finds the farthest point from the randomly picked point, x0.
    3 - Finds the farthest point from x0, x1.
    4 - Calculate the line vector between points x1 and x0. (x0 and x1 are two extremes in the dataset)
    5 - Project all points onto this line vector.
    6 - Calculate the median of the projections.
    7 - Split the points based on the median.
    8 - The points that have their projections less than or equal to the median goes to the left node, 
    others go to the right.

    To use for classification time, it stores the median and the normalized vector on each node.
    The median will be used as a reference to determine whether go to the right or to the left given a target point.
    The norm vector is used to avoid recalculating it for every target point.

    """
            
    def __init__(self, X, X_idx, leaf_size):
        """ Init method to construct the tree structure.

        Parameters
        ----------
        X : List[Any]
            Construction points.
        X_idx : List[Any]
            Indices of the X points in the training set.
        leaf_size : int
            Number of points in leaves.
        """

        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.left = self.right = None
        self.median = None
        self.vector_norm = None
        
        n = X.shape[0]
        if n <= leaf_size:
            self.create_leaf(X, X_idx)
            return
        

        rand_idx = np.random.randint(n)
        rand_point = X[rand_idx]
        
        # find the most distant point from the randomly picked point
        x0 = find_farthest_point(X, rand_point)
        
        # find the point most distant from the farthest point
        x1 = find_farthest_point(X, x0)

        # calculate the line vector between them
        self.line_vector = x1 - x0
        
        # project all points onto this line
        self.vector_norm = np.linalg.norm(self.line_vector)
        projections = np.dot(X, self.line_vector) / self.vector_norm
        
        # split based on the median of the projections
        self.median = np.median(projections)
        left_idx = projections <= self.median
        right_idx = projections > self.median
        
        self.left = BallTreeNode(X[left_idx], X_idx[left_idx], leaf_size)
        self.right = BallTreeNode(X[right_idx], X_idx[right_idx], leaf_size)
