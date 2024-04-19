import numpy as np
import scipy

from scipy.spatial import distance
from utils import divide_chunks, measure_execution_time


class KNN:
    def __init__(self, k = 3):
        self.k = k
        self._X_train = None
        self._classes = None
        self._y_train_indices = None
        self._sample_size = None

    def fit(self, X_train, y_train):
        self._X_train = np.array(X_train)
        self._target = np.array(y_train)
        y_train = np.array(y_train)
        if y_train.ndim != 1:
            raise ValueError("Dataset target must have only 1 dimension.")
        
        self._classes, indices = np.unique(y_train, return_inverse=True)
        self._sample_size = self._X_train.shape[0]
        self._y_train_indices = indices.reshape((-1, 1))

    
    # @staticmethod
    # def __euclidean_distances(X_train, X_test):
    #     return distance.cdist(X_test, X_train, 'euclidean')
        # return np.sqrt(np.sum(np.square(np.expand_dims(X_test, axis=1) - X_train), axis=2))
    
    def __compute_distances(self, X_test, n_chunks):
        distances = np.empty((X_test.shape[0], self.k), dtype=np.int32)

        for i, X_test_chunk in enumerate(divide_chunks(X_test, n_chunks)):
            start = i * n_chunks
            end = start + X_test_chunk.shape[0]
            dists = distance.cdist(X_test_chunk, self._X_train, 'euclidean')
            distances[start:end, :] = np.argsort(dists, axis=1)[:, :self.k]

        return distances

    @measure_execution_time
    def predict(self, X_test, n_chunks=256):
        _X_test = np.array(X_test)
        _classes = self._classes
        _y_train_idxs = self._y_train_indices

        distances = self.__compute_distances(_X_test, n_chunks)

        n_classes = len(_classes)
        n_X_test = len(_X_test)

        y_pred = np.empty((n_X_test, n_classes), dtype=_classes[0].dtype)
        mode = scipy.stats.mode(_y_train_idxs[distances], axis=1, keepdims=True).mode
        mode = np.asarray(np.ravel(mode))
        y_pred = _classes.take(mode)

        return np.ravel(y_pred)
    