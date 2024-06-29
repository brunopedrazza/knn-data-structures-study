from typing import Any

class Node:
    X: Any
    X_idx: Any
    is_leaf: bool
    right: 'Node'
    left: 'Node'

    def create_leaf(self, X, X_idx):
        """ Base method to create a leaf node with points an indices. Set node with is_leaf = True;

        Parameters
        ----------
        X : List[Any]
            Points to add to the leaf node.
        X_idx : List[Any]
            Indices of the X points in the training set.
        """

        self.X = X
        self.X_idx = X_idx
        self.is_leaf = True