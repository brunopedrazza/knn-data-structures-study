import numpy as np

from kd_tree import KdTree


if __name__ == "__main__":
    points2d = [
        [21,42],[10,100],[15,200],[20,50],[20,300]
    ]

    points3d = [
        [10,0,100],[5,10,15],[7,7,7],[9,0,8],[11,10,9],[11,0,0]
    ]

    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 3))

    points = np.array(X)
    
    kd_tree = KdTree()
    for p in points:
        kd_tree.add_node(p)

    kd_tree.display()