from sklearn import datasets
from sklearn.decomposition import PCA
from knn import KNN

import matplotlib.pyplot as plt
import numpy as np
import random_data


if __name__ == "__main__":
    dataset = datasets.load_iris()

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    new_point = random_data.generate_point_clustering(dataset.data)

    clf = KNN(k=3)
    clf.fit(dataset.data, dataset.target)
    prediction = clf.predict(new_point)
    prediction_name = dataset.target_names[prediction]

    print(f"{prediction}: {prediction_name}")

    prediction_data = np.vstack([dataset.data, new_point])

    X_reduced = PCA(n_components=3).fit_transform(dataset.data)

    prediction_reduced_point = PCA(n_components=3).fit_transform(prediction_data)[-1]

    # Define a colormap
    cmap = plt.cm.Set1

    # Use the colormap to generate colors
    colors = cmap(dataset.target / np.max(dataset.target))
    prediction_color = cmap(prediction / np.max(dataset.target))

    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=colors,
        s=40
    )
    ax.scatter(
        prediction_reduced_point[0],
        prediction_reduced_point[1],
        prediction_reduced_point[2],
        color=prediction_color,
        s=200,
        marker="*"
    )

    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("1st Eigenvector")
    # ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    # ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    # ax.zaxis.set_ticklabels([])

    plt.show()