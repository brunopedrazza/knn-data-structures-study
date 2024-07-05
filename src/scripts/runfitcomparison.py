import numpy as np
import matplotlib.pyplot as plt

from helpers.benchmark import bench_fit_duration


if __name__ == "__main__":
    n_features = 20
    n_samples_i = [100, 400, 1000, 4000, 6000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    methods = ["brute_force", "kd_tree", "ball_tree", "ball*_tree", "vp_tree"]
    methods_name = ["Força bruta", "KD-tree", "Ball-tree", "Ball*-tree", "VP-tree"]
    for i, method in enumerate(methods):
        results = []
        for n_samples in n_samples_i:
            X = np.random.rand(n_samples, n_features)
            y = np.random.rand(n_samples, 1)
            fit_time = bench_fit_duration(X, y, method, n_iterations=20)
            results.append((n_samples, fit_time))
        results = np.array(results)
        plt.plot(results[:, 0], results[:, 1], '-', label=methods_name[i])

    plt.legend()
    plt.ylabel("Tempo de treinamento (s)")
    plt.xlabel("# Instâncias")
    # plt.title(f"Fit Duration Comparison ({n_features} features)")
    plt.show()

