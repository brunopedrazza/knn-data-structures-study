import numpy as np

from trees.nodes.node import Node

def find_most_distant_point(X, point):
    distances = np.linalg.norm(X - point, axis=1)
    farthest_idx = np.argmax(distances)
    return X[farthest_idx]

class BallTreeNode(Node):

    def __init__(self, X, X_idx, leaf_size, depth=0):
        self.X = None
        self.X_idx = None
        self.is_leaf = False
        self.left = self.right = None
        self.median = None
        
        n = X.shape[0]
        if n <= leaf_size:
            self.create_leaf(X, X_idx)
            return
        
        rand_idx = np.random.randint(X.shape[0])
        rand_point = X[rand_idx]
        
        # Find the most distant point from the randomly picked point
        farthest_point = find_most_distant_point(X, rand_point)
        
        # Find the point most distant from the farthest point
        second_farthest_point = find_most_distant_point(X, farthest_point)

        # Calculate the line vector between them
        self.line_vector = second_farthest_point - farthest_point
        
        # Project all points onto this line
        projections = np.dot(X, self.line_vector) / np.linalg.norm(self.line_vector)
        
        # Split based on the median of the projections
        self.median = np.median(projections)
        left_idx = projections <= self.median
        right_idx = projections > self.median
        
        self.left = BallTreeNode(X[left_idx], X_idx[left_idx], leaf_size, depth+1)
        self.right = BallTreeNode(X[right_idx], X_idx[right_idx], leaf_size, depth+1)