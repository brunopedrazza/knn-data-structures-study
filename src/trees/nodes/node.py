class Node:
    
    def __init__(self, X, X_idx, leaf_size, depth): 
        ...

    def create_leaf(self, X, X_idx):
        self.X = X
        self.X_idx = X_idx
        self.is_leaf = True