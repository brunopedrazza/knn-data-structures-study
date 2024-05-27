from tabulate import tabulate
from metrics_collector import collect_metrics
from ucimlrepo.fetch import fetch_ucirepo

if __name__ == "__main__":
    k_start, k_end, k_step = (3, 7, 2)

    leaf_size = 100
    max_num_calls = 100
    methods = [
        "brute_force",
        "kd_tree",
        "ball_tree",
    ]

    database_ids = [
        53,     # Iris (very low instances (very low instances - 150, low dimensions - 4) - https://archive.ics.uci.edu/dataset/53/iris
        110,    # Yeast (low instances (low instances - 1484, low dimensions - 8) - https://archive.ics.uci.edu/dataset/110/yeast
        229,    # Skin Segmentation (high instances - 245057, low dimensions - 3) - https://archive.ics.uci.edu/dataset/229/skin+segmentation
        31,     # Covertype (high instances - 581012, high dimensions - 54) - https://archive.ics.uci.edu/dataset/31/covertype
        602,    # Dry Bean (medium instances - 13611, medium dimensions - 16) - https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
        372,    # HTRU2 (medium instances - 17898, low dimensions - 8) - https://archive.ics.uci.edu/dataset/372/htru2
        80,     # Optical Recognition of Handwritten Digits (low instances - 5620, high dimensions - 64) - https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
        891,    # CDC Diabetes Health Indicators (high instances - 253680, medium dimensions - 21) - https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
        78,     # Page Blocks Classification (medium instances - 5473, medium dimensions - 10) - https://archive.ics.uci.edu/dataset/78/page+blocks+classification
        545,    # Rice (Cammeo and Osmancik) (medium instances - 3810, medium dimensions - 7) - https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
        159,    # MAGIC Gamma Telescope (medium instances - 19020, medium dimensions - 10) - https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
        148,    # Statlog (Shuttle) (high instances - 58000, medium dimensions - 7) - https://archive.ics.uci.edu/dataset/148/statlog+shuttle
    ]
    
    for db_id in database_ids:
        results = []
        db = fetch_ucirepo(id=db_id)
        X, y = db.data.features.values, db.data.targets.values
        print()
        print(f"{db.metadata.name}, n_samples={len(X)}, n_dimensions={len(X[0])}")
        num_calls = 1 if len(X) > 5_000 else max_num_calls
        for k in range(k_start, k_end+1, k_step):
            for method in methods:
                result = collect_metrics(X, y, k, method, leaf_size, num_calls=num_calls)
                results.append(result)
        for result in results:
            improv = 0
            if result["method"] != "brute_force":
                brute_predict_duration = next((item["predict_duration_seconds"] for item in results if item["method"] == "brute_force" and item["k"] == result["k"]), 0)
                # improv = (1 - (result["predict_duration_seconds"]/brute_predict_duration)) * 100 if brute_predict_duration != 0 else 0
                improv = result["predict_duration_seconds"] - brute_predict_duration if brute_predict_duration != 0 else 0
            # result["predict_performance_improv_%"] = math.trunc(improv)
            result["predict_diff_seconds"] = improv
        print(tabulate(results, headers="keys", tablefmt="rounded_grid"))

