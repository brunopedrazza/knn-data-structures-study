import matplotlib.pyplot as plt
import numpy as np

from kd_tree import KdTree
from knn import KNN

if __name__ == "__main__":
    points = [
        [2,4],[1,3],[2,5],[3,2],[2,1],
        [5,6],[4,5],[3,6],[6,6],[5,4],[1,4]
    ]
    targets = [
        "blue", "blue", "blue", "blue", "blue",
        "red", "red", "red", "red", "red", "blue"
    ]
    
    kd_tree = KdTree.construct(points, targets)

    kd_tree.display()
    
    new_points = [[2,3],[4,4]]

    new_classes = list(kd_tree.predict(new_points))

    # new_classes = list(kd_tree.predict(new_points))
    
    # clf = KNN(k=3)
    # clf.fit(points, targets)
    # new_classes = clf.predict(new_points)
    # print(new_classes)

    # Visualize
    
    grey = "#7d7979"
    black = "#121212"
    blue = "#104DCA"
    red = "#FF0000"

    ax = plt.subplot()
    ax.grid(True, color=grey)
    ax.tick_params(axis="x", color="white")
    ax.tick_params(axis="y", color="white")

    def t(kd_tree):
        if not kd_tree:
            return
        point_color = blue if kd_tree.class_ == "blue" else red
        ax.scatter(kd_tree.point[0], kd_tree.point[1], color=point_color, s=60)
        t(kd_tree.left)
        t(kd_tree.right)

    t(kd_tree)

    for index, new_class in enumerate(new_classes):
        new_point_color = blue if new_class == "blue" else red
        ax.scatter(new_points[index][0], new_points[index][1], color=new_point_color, marker="*", s=200, zorder=100)

    plt.show()