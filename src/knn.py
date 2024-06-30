import numpy as np
import scipy

from trees.vptree import VpTree
from trees.kdtree import KdTree
from trees.ballwtree import BallWTree
from trees.balltree import BallTree


class KNN:
    """ K-nearest neighbors classifier.

    Parameters
    ----------
    k : int, default=3
        Number of neighbors to use by default.

    method : {'brute_force', 'kd_tree', 'ball_tree', 'ball*_tree', 'vp_tree'}, default='brute_force'
        Method used to compute the nearest neighbors:

        - 'brute_force' will use a brute-force search.
        - 'kd_tree' will use :class:`KdTree`
        - 'ball_tree' will use :class:`BallTree`
        - 'ball*_tree' will use :class:`BallWTree`
        - 'vp_tree' will use :class:`VpTree`
    
    leaf_size : int, default=30
        Leaf size passed to BallTree, KdTree or VpTree.
    """

    def __init__(self, k=3, method="brute_force", leaf_size=30):
        if method not in ("brute_force", "kd_tree", "ball_tree", "ball*_tree", "vp_tree"):
            raise ValueError("Invalid method")
        if leaf_size <= 1:
            raise ValueError("Invalid leaf size")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be strictly positive integer")
        self._k = k
        self._method = method
        self._tree = None
        self._leaf_size = leaf_size
        self._X_train = None
        self._classes = None
        self._y_train_indices = None
        self._sample_size = None

    def fit(self, X_train, y_train):
        """ Method that fits training data. It will use the tree structures or plain storing.

        Parameters
        ----------
        X_train : array
            Training data.
        
        y_train : array
            Classification of the training data.
        """

        y_train = np.array(y_train)
        X_train = np.array(X_train)

        y_train = np.ravel(y_train)

        if y_train.ndim != 1:
            raise ValueError("Dataset target must have only 1 dimension.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Training data and targets must have the same size.")
        
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
        elif self._method == "ball_tree":
            self._tree = BallTree(X_train, k=self._k, leaf_size=self._leaf_size)
        elif self._method == "ball*_tree":
            self._tree = BallWTree(X_train, k=self._k, leaf_size=self._leaf_size)
        elif self._method == "vp_tree":
            self._tree = VpTree(X_train, k=self._k, leaf_size=self._leaf_size)
    
    def __compute_distances(self, X_test):
        """ Internal method used for brute force. It iterates through all test points, 
        calculate all distances, sorts it and store the k nearest neighbors.
        """

        best_idxs = np.empty((X_test.shape[0], self._k), dtype=np.int32)

        for i, target in enumerate(X_test):
            dists = np.linalg.norm(self._X_train - target, axis=1)
            best_idxs[i, :] = np.argsort(dists)[:self._k]

        return best_idxs

    def predict(self, X_test):
        """ Predict the classes for the testing set. Uses different mehods to do so, 
        according to what was used to fit.

        Parameters
        ----------
        X_test : array
            Training data.
        """

        _X_test = np.array(X_test)
        _classes = self._classes
        _y_train_idxs = self._y_train_indices
        n_classes = len(_classes)
        n_X_test = len(_X_test)

        avg_nodes_visited = 0
        max_depth = 0
        if self._method == "brute_force":
            best_idxs = self.__compute_distances(_X_test)
        elif self._method in ("kd_tree", "ball_tree", "ball*_tree", "vp_tree"):
            best_idxs = self._tree.predict(_X_test)
            avg_nodes_visited = self._tree.total_points_visited / n_X_test
            max_depth = self._tree.max_depth


        y_pred = np.empty((n_X_test, n_classes), dtype=_classes.dtype)
        mode = scipy.stats.mode(_y_train_idxs[best_idxs], axis=1, keepdims=True).mode
        mode = np.asarray(np.ravel(mode))
        y_pred = _classes.take(mode)

        return np.ravel(y_pred), avg_nodes_visited, max_depth
    