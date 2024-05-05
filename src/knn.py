import numpy as np
import scipy

from kd_tree import KdTree
from utils import divide_chunks, measure_execution_time, euclidean_distance


class KNN:
    def __init__(self, k = 3, method="brute_force"):
        if method not in ["brute_force", "kd_tree"]:
            raise ValueError("Invalid method")
        self._k = k
        self._method = method
        self._tree = None
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
            self._tree = KdTree.construct(X_train)

    
    # @staticmethod
    # def __euclidean_distances(X_train, X_test):
    #     return distance.cdist(X_test, X_train, 'euclidean')
        # return np.sqrt(np.sum(np.square(np.expand_dims(X_test, axis=1) - X_train), axis=2))
    
    def __compute_distances(self, X_test, size_chunks):
        best_idxs = np.empty((X_test.shape[0], self._k), dtype=np.int32)

        for i, X_test_chunk in enumerate(divide_chunks(X_test, size_chunks)):
            start = i * size_chunks
            end = start + X_test_chunk.shape[0]
            dists = euclidean_distance(X_test_chunk, self._X_train)
            best_idxs[start:end, :] = np.argsort(dists, axis=1)[:, :self._k]

        return best_idxs

    @measure_execution_time
    def predict(self, X_test, size_chunks=256):
        _X_test = np.array(X_test)
        _classes = self._classes
        _y_train_idxs = self._y_train_indices

        if self._method == "brute_force":
            best_idxs = self.__compute_distances(_X_test, size_chunks)
        elif self._method == "kd_tree":
            best_idxs = self._tree.predict(_X_test, k=self._k)

        n_classes = len(_classes)
        n_X_test = len(_X_test)

        y_pred = np.empty((n_X_test, n_classes), dtype=_classes[0].dtype)
        mode = scipy.stats.mode(_y_train_idxs[best_idxs], axis=1, keepdims=True).mode
        mode = np.asarray(np.ravel(mode))
        y_pred = _classes.take(mode)

        return np.ravel(y_pred)
    