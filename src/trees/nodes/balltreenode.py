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
        self.centroid = None
        self.radius = None
        
        self.centroid = np.mean(X, axis=0)
        pleft = find_farthest_point(X, self.centroid)
        self.radius = np.linalg.norm(self.centroid - pleft, axis=0)

        n = X.shape[0]

        if n <= leaf_size:
            self.create_leaf(X, X_idx)
            return

        pright = find_farthest_point(X, pleft)
        distances_to_pleft = np.linalg.norm(X - pleft, axis=1)
        distances_to_pright = np.linalg.norm(X - pright, axis=1)

        belongs_left = distances_to_pleft < distances_to_pright

        self.left = BallTreeNode(X[belongs_left], X_idx[belongs_left], leaf_size)
        self.right = BallTreeNode(X[~belongs_left], X_idx[~belongs_left], leaf_size)
