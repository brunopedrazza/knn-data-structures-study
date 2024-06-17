import numpy as np

from trees.nodes.node import Node


class VpTreeNode(Node):
    """ Class that defines a VpTree node. Used to construct a VpTree structure.

    Steps to construct the tree:
    1 - Choose the vantage point given the index.
    2 - Calculate the distances between the vantage point and the other ones
    3 - Calculate the median of the distances and set it as the threshold.
    4 - Split the points that are below and above the thresholf defined.
    5 - Get the index of the farthest point from the vantage point on each child, 
    they will be used as the index of the vantage point for each child.

    To use for classification time, it stores the vantage point and the threshold on each node.

    """

    def __init__(self, X, X_idx, leaf_size, vp_idx=None):
        """ Init method to construct the tree structure.

        Parameters
        ----------
        X : List[Any]
            Construction points.
        X_idx : List[Any]
            Indices of the X points in the training set.
        leaf_size : int
            Number of points in leaves.
        vp_idx : int
            Index of the vantage point.
        """

        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.closer = self.farther = None
        self.vp = None
        self.t = None

        n = X.shape[0]

        if n <= leaf_size:
            self.create_leaf(X, X_idx)
            return

        vp_idx = 0 if vp_idx is None else vp_idx
        self.vp = X[vp_idx]

        # Choose division boundary at median of distances
        distances = np.linalg.norm(X - self.vp, axis=1)

        # Compute the median of distances
        self.t = np.median(distances)

        X_closer = []
        X_farther = []
        X_idx_closer = []
        X_idx_farther = []
        closer_max_d = 0
        closer_max_d_idx = None
        farther_max_d = 0
        farther_max_d_idx = None
        for point, idx, distance in zip(X, X_idx, distances):
            if distance < self.t:
                X_closer.append(point)
                X_idx_closer.append(idx)
                if distance > closer_max_d:
                    closer_max_d = distance
                    closer_max_d_idx = len(X_closer) - 1
            else:
                X_farther.append(point)
                X_idx_farther.append(idx)
                if distance > farther_max_d:
                    farther_max_d = distance
                    farther_max_d_idx = len(X_farther) - 1

        X_closer = np.array(X_closer)
        X_farther = np.array(X_farther)
        X_idx_closer = np.array(X_idx_closer)
        X_idx_farther = np.array(X_idx_farther)
        if len(X_closer) > 0:
            self.closer = VpTreeNode(X_closer, X_idx_closer, leaf_size, vp_idx=closer_max_d_idx)

        if len(X_farther) > 0:
            self.farther = VpTreeNode(X_farther, X_idx_farther, leaf_size, vp_idx=farther_max_d_idx)
