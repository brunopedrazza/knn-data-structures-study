import math
import numpy as np

from helpers.utils import euclidean_distance
from trees.nodes.node import Node

class VpTreeNode(Node):
    
    def __init__(self, X, X_idx, leaf_size, vp_idx=None):

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

        vp_idx = np.random.randint(X.shape[0]) if vp_idx is None else vp_idx
        self.vp = X[vp_idx]

        # Choose division boundary at median of distances.
        distances = euclidean_distance(np.array([self.vp]), X)[0]

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