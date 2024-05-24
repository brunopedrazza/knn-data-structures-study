from sklearn import datasets as ds
import math
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier

from metrics_collector import collect_metrics

if __name__ == "__main__":
    k_start, k_end, k_step = (3, 7, 2)

    leaf_size = 100
    methods = ["brute_force", "kd_tree", "ball_tree"]
    # methods = ["kd_tree", "ball_tree"]
    # methods = ["ball_tree"]
    database_opts = [
        ds.load_breast_cancer,
        ds.load_digits,
        ds.load_iris,
        ds.load_wine,
        ds.fetch_olivetti_faces,
        ds.fetch_covtype
    ]
    
    for db_method in database_opts:
        results = []
        X, y = db_method(return_X_y=True)
        print()
        print(f"{db_method.__name__.partition("_")[2]}, n_samples={len(X)}, n_dimensions={len(X[0])}")
        for k in range(k_start, k_end+1, k_step):
            for method in methods:
                result = collect_metrics(X, y, k, method, leaf_size)
                results.append(result)
        for result in results:
            improv = 0
            if result["method"] != "brute_force":
                brute_predict_duration = next(item["predict_duration_seconds"] for item in results if item["method"] == "brute_force" and item["k"] == result["k"])
                improv = (1 - (result["predict_duration_seconds"]/brute_predict_duration)) * 100
            result["predict_performance_improv"] = f"{math.trunc(improv)}%"
        print(tabulate(results, headers="keys", tablefmt="rounded_grid"))

