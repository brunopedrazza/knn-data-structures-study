from typing import Any
from trees.nodes.node import Node


class BallTreeNodeBase(Node):
    centroid: Any
    radius: Any
    right: 'BallTreeNodeBase'
    left: 'BallTreeNodeBase'
