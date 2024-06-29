import numpy as np
from numpy.linalg import eigh

from helpers.utils import find_farthest_point
from trees.nodes.balltreenodebase import BallTreeNodeBase


class BallWTreeNode(BallTreeNodeBase):
    """ Class that defines a BallWTree node. Used to construct a BallWTree structure.

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
        
        self.centroid = np.mean(X, axis=0)
        p = find_farthest_point(X, self.centroid)
        self.radius = np.linalg.norm(self.centroid - p, axis=0)
        
        n = X.shape[0]
        if n <= leaf_size:
            self.create_leaf(X, X_idx)
            return
        
        # centralize the data
        data_centered = X - self.centroid
    
        # calculate the covariance matrix
        covariance_matrix = np.cov(data_centered, rowvar=False)
        
        # calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        
        # sort the eigenvectors by the eigenvalues in descending order
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_index]
        
        # the principal component is the eigenvector with the highest eigenvalue
        principal_component = sorted_eigenvectors[:, 0]

        # project all points onto this line
        projections = np.dot(X, principal_component) / np.linalg.norm(principal_component)
        
        # split based on the median of the projections
        median = np.median(projections)
        left_idx = projections <= median
        right_idx = projections > median
        
        self.left = BallWTreeNode(X[left_idx], X_idx[left_idx], leaf_size)
        self.right = BallWTreeNode(X[right_idx], X_idx[right_idx], leaf_size)
