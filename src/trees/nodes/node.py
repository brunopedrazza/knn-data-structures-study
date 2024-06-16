class Node:
    
    def __init__(self, X, X_idx, leaf_size: int, depth: int): 
        """ Recursively constructs a tree structure.

        Parameters
        ----------
        X : List[Any]
            Construction points.
        X_idx : List[Any]
            Indices of the X points in the training set.
        leaf_size : int
            Number of points in leaves.
        depth : int
            Depth of that node related to the tree structure
        """
        ...


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