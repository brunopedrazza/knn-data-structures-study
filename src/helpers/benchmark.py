from time import perf_counter
from sklearn.model_selection import train_test_split
from knn import KNN

import sklearn.metrics as metrics


def benchmark(X, y, k, method, leaf_size=None, n_iterations=100):
    if not leaf_size and method not in ("kd_tree", "kd_tree_opt", "ball_tree"):
        raise ValueError("Leaf size is required when method is tree based")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    fit_duration_seconds = 0
    predict_duration_seconds = 0
    for _ in range(0, n_iterations):
        knn = KNN(k=k, method=method, leaf_size=leaf_size)

        tic = perf_counter()
        knn.fit(X_train, y_train)
        toc = perf_counter()
        fit_duration_seconds += (toc - tic)

        tic = perf_counter()
        y_pred, avg_nodes_visited, max_depth = knn.predict(X_test)
        toc = perf_counter()
        predict_duration_seconds += (toc - tic)

    acc = metrics.accuracy_score(y_test, y_pred)
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fit_duration_seconds": fit_duration_seconds / n_iterations,
        "predict_duration_seconds": predict_duration_seconds / n_iterations,
        "accuracy": acc,
        "avg_nodes_visited": avg_nodes_visited,
        "max_depth": max_depth
    }

def bench_fit_duration(X, y, method, n_iterations=100):
    fit_duration_seconds = 0
    for _ in range(0, n_iterations):
        knn = KNN(k=1, method=method, leaf_size=2)

        tic = perf_counter()
        knn.fit(X, y)
        toc = perf_counter()
        fit_duration_seconds += (toc - tic)

    return fit_duration_seconds / n_iterations

def bench_predict_duration(X, y, X_test, k, method, n_iterations=100, leaf_size=2):
    predict_duration_seconds = 0
    knn = KNN(k=k, method=method, leaf_size=leaf_size)
    knn.fit(X, y)
    for _ in range(0, n_iterations):
        tic = perf_counter()
        knn.predict(X_test)
        toc = perf_counter()
        predict_duration_seconds += (toc - tic)

    return predict_duration_seconds / n_iterations
