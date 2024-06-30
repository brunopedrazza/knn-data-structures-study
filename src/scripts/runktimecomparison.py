import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from helpers.benchmark import bench_predict_duration
from ucimlrepo.fetch import fetch_ucirepo


if __name__ == "__main__":
    methods = ["brute_force", "ball_tree"]
    dataset = fetch_ucirepo(id=545)

    X, y = np.array(dataset.data.features.values), np.array(dataset.data.targets.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    for method in methods:
        results = []
        for k in range(1, 60, 2):
            predict_time = bench_predict_duration(X_train, y_train, X_test, k, method, n_iterations=20, leaf_size=100)
            results.append((k, predict_time))
        results = np.array(results)
        plt.plot(results[:, 0], results[:, 1], '-', label=method)

    plt.legend()
    plt.ylabel("Time (sec)")
    plt.xlabel("k")
    plt.title(f"K Time Comparison")
    plt.show()

