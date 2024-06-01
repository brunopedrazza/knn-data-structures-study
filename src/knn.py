import numpy as np
import scipy

from trees.kdtree import KdTree
from trees.balltree import BallTree
from helpers.utils import euclidean_distance


class KNN:
    
    def __init__(self, k = 3, method="brute_force", leaf_size=30):
        if method not in ("brute_force", "kd_tree", "kd_tree_opt", "ball_tree"):
            raise ValueError("Invalid method")
        if leaf_size <= 1:
            raise ValueError("Invalid leaf size")
        self._k = k
        self._method = method
        self._tree = None
        self._leaf_size = leaf_size
        self._X_train = None
        self._classes = None
        self._y_train_indices = None
        self._sample_size = None

    def fit(self, X_train, y_train):
        y_train = np.array(y_train)
        X_train = np.array(X_train)

        y_train = np.ravel(y_train)

        if y_train.ndim != 1:
            raise ValueError("Dataset target must have only 1 dimension.")
        
        # remove duplicated points
        _, unique_indices = np.unique(X_train, axis=0, return_index=True)
        X_train = X_train[unique_indices]
        y_train = y_train[unique_indices]

        self._target = y_train
        
        self._classes, indices = np.unique(y_train, return_inverse=True)
        self._sample_size = len(X_train)
        self._y_train_indices = indices.reshape((-1, 1))
        
        if self._method == "brute_force":
            self._X_train = np.array(X_train)
        elif self._method == "kd_tree":
            self._tree = KdTree(X_train, k=self._k, leaf_size=self._leaf_size)
        elif self._method == "kd_tree_opt":
            self._tree = KdTree(X_train, k=self._k, leaf_size=self._leaf_size, optimized=True)
        elif self._method == "ball_tree":
            self._tree = BallTree(X_train, k=self._k, leaf_size=self._leaf_size)
    
    def __compute_distances(self, X_test):
        best_idxs = np.empty((X_test.shape[0], self._k), dtype=np.int32)

        for i, target in enumerate(X_test):
            dists = euclidean_distance(np.array([target]), self._X_train)
            best_idxs[i, :] = np.argsort(dists, axis=1)[:, :self._k]

        return best_idxs

    def predict(self, X_test):
        _X_test = np.array(X_test)
        _classes = self._classes
        _y_train_idxs = self._y_train_indices

        distance_count = None
        if self._method == "brute_force":
            best_idxs = self.__compute_distances(_X_test)
            distance_count = self._X_train.shape[0] * _X_test.shape[0]
        elif self._method in ("kd_tree", "kd_tree_opt", "ball_tree"):
            best_idxs = self._tree.predict(_X_test)
            distance_count = self._tree.distance_count

        n_classes = len(_classes)
        n_X_test = len(_X_test)

        y_pred = np.empty((n_X_test, n_classes), dtype=_classes.dtype)
        mode = scipy.stats.mode(_y_train_idxs[best_idxs], axis=1, keepdims=True).mode
        mode = np.asarray(np.ravel(mode))
        y_pred = _classes.take(mode)

        return np.ravel(y_pred), distance_count
    