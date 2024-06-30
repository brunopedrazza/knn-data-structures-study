import numpy as np

from helpers.utils import find_farthest_point
from trees.nodes.node import Node


class BallTreeNode(Node):
    """ Class that defines a BallTree node. Used to construct a BallTree structure.

    Steps to construct the tree:
    1 - Calculate the centroid based on the mean of X.
    2 - Find the farthest point from the centroid, this will be our left pivot.
    3 - Calculate the radius based on the distance to the left pivot.
    4 - Find the farthest point from the left pivot, this will be our right pivot.
    7 - Split the points based on the proximity with left pivot and right pivot.
    8 - The points closer to the left eft pivot goes to the left child, and other to the right.

    To use for classification time, it stores the centroid and radius on each node.
    The centroid will be used as a reference to determine whether go to the right or to the left given a target point.
    The radius is used to check if whe can prune the other child.

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
