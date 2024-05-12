import numpy as np
import scipy

from kd_tree import KdTree
from kd_tree_leaf import KdTreeLeaf
from utils import measure_execution_time, euclidean_distance


class KNN:
    def __init__(self, k = 3, method="brute_force", leaf_size=30):
        if method not in ("brute_force", "kd_tree", "kd_tree_leaf"):
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

    @measure_execution_time
    def fit(self, X_train, y_train):
        y_train = np.array(y_train)
        if y_train.ndim != 1:
            raise ValueError("Dataset target must have only 1 dimension.")
        
        self._target = np.array(y_train)
        
        self._classes, indices = np.unique(y_train, return_inverse=True)
        self._sample_size = len(X_train)
        self._y_train_indices = indices.reshape((-1, 1))
        
        if self._method == "brute_force":
            self._X_train = np.array(X_train)
        elif self._method == "kd_tree":
            self._tree = KdTree(X_train)
        elif self._method == "kd_tree_leaf":
            self._tree = KdTreeLeaf(X_train, self._k, self._leaf_size)
    
    def __compute_distances(self, X_test):
        best_idxs = np.empty((X_test.shape[0], self._k), dtype=np.int32)

        for i, target in enumerate(X_test):
            dists = np.empty((self._X_train.shape[0]), dtype=np.float64)
            for j, x in enumerate(self._X_train):
                dists[j] = euclidean_distance(x, target, squared=True)
            best_idxs[i] = np.argsort(dists)[:self._k]

        return best_idxs
    
    def __compute_distances2(self, X_test):
        best_idxs = np.empty((X_test.shape[0], self._k), dtype=np.int32)

        for i, target in enumerate(X_test):
            dists = euclidean_distance(np.array([target]), self._X_train)
            best_idxs[i, :] = np.argsort(dists, axis=1)[:, :self._k]

        return best_idxs

    @measure_execution_time
    def predict(self, X_test):
        _X_test = np.array(X_test)
        _classes = self._classes
        _y_train_idxs = self._y_train_indices

        distance_count = None
        if self._method == "brute_force":
            best_idxs = self.__compute_distances2(_X_test)
            distance_count = self._X_train.shape[0] * _X_test.shape[0]
        elif self._method in ("kd_tree", "kd_tree_leaf"):
            best_idxs = self._tree.predict2(_X_test, k=self._k)
            distance_count = self._tree.distance_count

        n_classes = len(_classes)
        n_X_test = len(_X_test)

        y_pred = np.empty((n_X_test, n_classes), dtype=_classes[0].dtype)
        mode = scipy.stats.mode(_y_train_idxs[best_idxs], axis=1, keepdims=True).mode
        mode = np.asarray(np.ravel(mode))
        y_pred = _classes.take(mode)

        return np.ravel(y_pred), distance_count
    