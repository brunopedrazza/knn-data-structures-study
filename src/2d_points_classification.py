import matplotlib.pyplot as plt

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
    
    new_points = [[2,3],[4,4]]
    
    clf = KNN(k=11, method="kd_tree")
    clf.fit(points, targets)
    new_classes = clf.predict(new_points)
    print(new_classes)

    # Visualize
    
    grey = "#7d7979"
    black = "#121212"
    blue = "#104DCA"
    red = "#FF0000"

    ax = plt.subplot()
    ax.grid(True, color=grey)
    ax.tick_params(axis="x", color="white")
    ax.tick_params(axis="y", color="white")

    for index, p in enumerate(points):
        point_color = blue if targets[index] == "blue" else red
        ax.scatter(p[0], p[1], color=point_color, s=60)

    for index, new_class in enumerate(new_classes):
        new_point_color = blue if new_class == "blue" else red
        ax.scatter(new_points[index][0], new_points[index][1], color=new_point_color, marker="*", s=200, zorder=100)

    plt.show()