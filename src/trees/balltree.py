from trees.balltreebase import BallTreeBase
from trees.nodes.balltreenode import BallTreeNode


class BallTree(BallTreeBase):

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, BallTreeNode)
