import numpy as np
import matplotlib.pyplot as plt

from helpers.benchmark import benchmark
from ucimlrepo.fetch import fetch_ucirepo


if __name__ == "__main__":
    method = "ball_tree"
    dataset = fetch_ucirepo(id=545)

    X, y = np.array(dataset.data.features.values), np.array(dataset.data.targets.values)

    results = []
    for k in range(1, 90, 1):
        beanch = benchmark(X, y, k, method, n_iterations=1, leaf_size=100)
        results.append((k, beanch["accuracy"]))
    results = np.array(results)
    plt.plot(results[:, 0], results[:, 1], '-')

    plt.legend()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("k")
    plt.title(f"K Accuracy Comparison")
    plt.show()

