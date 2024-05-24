from time import perf_counter
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from knn import KNN

def collect_metrics(X, y, k, method, leaf_size=None):
    if not leaf_size and method not in ("kd_tree", "ball_tree"):
        raise ValueError("Leaf size is required when method is tree based")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNN(k=k, method=method, leaf_size=leaf_size)

    tic = perf_counter()
    knn.fit(X_train, y_train)
    toc = perf_counter()
    fit_duration_seconds = toc - tic

    tic = perf_counter()
    y_pred, distance_count = knn.predict(X_test)
    toc = perf_counter()
    predict_duration_seconds = toc - tic

    acc = metrics.accuracy_score(y_test, y_pred)
    return {
        "k": k,
        "method": method,
        "fit_duration_ms": fit_duration_seconds * 1000,
        "predict_duration_seconds": predict_duration_seconds,
        "accuracy": acc,
        "distance_count": distance_count
    }