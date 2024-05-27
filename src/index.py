import csv
import os

from datetime import datetime
from tabulate import tabulate
from metrics_collector import collect_metrics
from ucimlrepo.fetch import fetch_ucirepo

def save_results_csv(results):
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    file_name = f"results_{date_str}.csv"

    current_directory = os.getcwd()
    directory = os.path.join(current_directory, "assets/results")
    file_path = os.path.join(directory, file_name)

    # check if the directory exists
    if not os.path.exists(directory):
        # create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)
    
    keys = results[0].keys()
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys, delimiter=";")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    k_start, k_end, k_step = (3, 7, 2)

    leaf_size = 100
    max_num_calls = 5
    methods = [
        "brute_force",
        "kd_tree",
        "ball_tree",
    ]

    database_ids = [
        53,     # Iris (very low instances (very low instances - 150, low dimensions - 4) - https://archive.ics.uci.edu/dataset/53/iris
        110,    # Yeast (low instances (low instances - 1484, low dimensions - 8) - https://archive.ics.uci.edu/dataset/110/yeast
        # 229,    # Skin Segmentation (high instances - 245057, low dimensions - 3) - https://archive.ics.uci.edu/dataset/229/skin+segmentation
        # 31,     # Covertype (high instances - 581012, high dimensions - 54) - https://archive.ics.uci.edu/dataset/31/covertype
        # 602,    # Dry Bean (medium instances - 13611, medium dimensions - 16) - https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
        # 372,    # HTRU2 (medium instances - 17898, low dimensions - 8) - https://archive.ics.uci.edu/dataset/372/htru2
        # 80,     # Optical Recognition of Handwritten Digits (low instances - 5620, high dimensions - 64) - https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
        # 891,    # CDC Diabetes Health Indicators (high instances - 253680, medium dimensions - 21) - https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
        # 78,     # Page Blocks Classification (medium instances - 5473, medium dimensions - 10) - https://archive.ics.uci.edu/dataset/78/page+blocks+classification
        # 545,    # Rice (Cammeo and Osmancik) (medium instances - 3810, medium dimensions - 7) - https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
        # 159,    # MAGIC Gamma Telescope (medium instances - 19020, medium dimensions - 10) - https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
        # 148,    # Statlog (Shuttle) (high instances - 58000, medium dimensions - 7) - https://archive.ics.uci.edu/dataset/148/statlog+shuttle
    ]
    
    dbs_data = []
    for db_id in database_ids:
        dataset = fetch_ucirepo(id=db_id)
        X, y = dataset.data.features.values, dataset.data.targets.values
        dbs_data.append((X, y, dataset.metadata))
    
    results = []
    for db_data in dbs_data:
        X, y, metadata = db_data
        num_calls = 1 if len(X) > 20_000 else max_num_calls
        for k in range(k_start, k_end+1, k_step):
            for method in methods:
                data = {
                    "db_id": metadata.uci_id,
                    "db_name": metadata.name,
                    "n_samples": len(X),
                    "n_dimensions": len(X[0]),
                    "k": k,
                    "method": method
                }
                result = collect_metrics(X, y, k, method, leaf_size, num_calls=num_calls)
                data.update(result)
                results.append(data)
    
    save_results_csv(results)
    
    # print(tabulate(results, headers="keys", tablefmt="rounded_grid"))
    # print()
    # print(f"{metadata.name}, n_samples={len(X)}, n_dimensions={len(X[0])}")
    # for result in results:
    #     improv = 0
    #     if result["method"] != "brute_force":
    #         brute_predict_duration = next((item["predict_duration_seconds"] for item in results if item["method"] == "brute_force" and item["k"] == result["k"]), 0)
    #         improv = result["predict_duration_seconds"] - brute_predict_duration if brute_predict_duration != 0 else 0
    #     result["predict_diff_seconds"] = improv
    # print(tabulate(results, headers="keys", tablefmt="rounded_grid"))
    # t = f"{acc*100:.2f}%"
    # "fit_duration_ms": (fit_duration_seconds / num_calls) * 1000,
