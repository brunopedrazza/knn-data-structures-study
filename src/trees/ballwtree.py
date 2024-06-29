from trees.balltreebase import BallTreeBase
from trees.nodes.ballwtreenode import BallWTreeNode


class BallWTree(BallTreeBase):

    """ Class that defines a BallWTree structure. Used to classify a target point.
    
    To traverse the tree recursively, the target point is projected on the line vector and it goes to the left if the projection
    is less than or equal to the current median or to the right if not. It keeps going to the "good" child until it reaches a leaf node.
    After going all the way to the "good" side, it have to check if it needs to check the "bad" side as well. If the distance 
    to the hyperplane is less than the distance of most distant neighbor found, it needs to check for the "bad" side.

    The closest neighbors are stored in a max heap structure. It is populated when it reaches a leaf node.
    """

    def __init__(self, X, k, leaf_size):
        super().__init__(X, k, leaf_size, BallWTreeNode)
