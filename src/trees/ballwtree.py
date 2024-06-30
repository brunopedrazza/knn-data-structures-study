from trees.balltreebase import BallTreeBase
from trees.nodes.ballwtreenode import BallWTreeNode


class BallWTree(BallTreeBase):

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, BallWTreeNode)
