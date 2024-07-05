import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from helpers.benchmark import bench_predict_duration
from ucimlrepo.fetch import fetch_ucirepo


if __name__ == "__main__":
    k = 3
    n_samples_i = [100, 1000, 4000, 6000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    # n_samples_i = [100, 1000, 4000, 6000, 10000, 15000, 20000, 30000, 40000]
    num_calls = [50, 50, 50, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    num_leaf_sizes = [5, 10, 30, 40, 60, 80, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    methods = ["kd_tree", "ball_tree", "ball*_tree", "vp_tree"]
    methods_name = ["KD-tree", "Ball-tree", "Ball*-tree", "VP-tree"]
    # methods = ["brute_force", "kd_tree", "ball_tree", "vp_tree"]

    dataset = fetch_ucirepo(id=31)
    X, y = np.array(dataset.data.features.values), np.array(dataset.data.targets.values)
    
    for j, method in enumerate(methods):
        results = []
        for i, n_samples in enumerate(n_samples_i):
            Xi = X[:n_samples]
            yi = y[:n_samples]
            X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=0.2, shuffle=True, random_state=42)
            predict_time = bench_predict_duration(X_train, y_train, X_test, k, method, n_iterations=num_calls[i], leaf_size=num_leaf_sizes[i])
            results.append((n_samples, predict_time))
        results = np.array(results)
        plt.plot(results[:, 0], results[:, 1], '-', label=methods_name[j])

    plt.legend()
    plt.ylabel("Tempo de classificação (s)")
    plt.xlabel("# Instâncias")
    # plt.title(f"Predict Duration Comparison ({X.shape[1]} dimensions)")
    plt.show()

