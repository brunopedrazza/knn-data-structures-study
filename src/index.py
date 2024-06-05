import csv
import os
import sys

from datetime import datetime
from metricscollector import collect_metrics
from showresults import print_results
from ucimlrepo.fetch import fetch_ucirepo
from helpers.utils import get_results_directory

def save_results_csv(results):
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    file_name = f"results_{date_str}.csv"

    result_directory = get_results_directory()
    file_path = os.path.join(result_directory, file_name)

    if not os.path.exists(result_directory):
        os.makedirs(result_directory, exist_ok=True)
    
    keys = results[0].keys()
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys, delimiter=";")
        writer.writeheader()
        writer.writerows(results)
    
    return file_name


if __name__ == "__main__":

    option_err_message = "Please choose whether you want to save the results or print then, with --save or --print"
    try:
        option = sys.argv[1]
        if option != "--save" and option != "--print":
            raise ValueError()
    except IndexError:
        print(option_err_message)
        exit()
    except ValueError:
        print(option_err_message)
        exit()

    k_start, k_end, k_step = (1, 1, 2)

    leaf_size = 100
    max_num_calls = 5
    methods = [
        # "brute_force",
        # "kd_tree",
        "kd_tree_opt",
        "ball_tree",
        "vp_tree",
        "vp_tree2"
    ]

    database_ids = [
        # 53,     # Iris (very low instances (very low instances - 150, low dimensions - 4) - https://archive.ics.uci.edu/dataset/53/iris
        # 110,    # Yeast (low instances (low instances - 1484, low dimensions - 8) - https://archive.ics.uci.edu/dataset/110/yeast
        # 229,    # Skin Segmentation (high instances - 245057, low dimensions - 3) - https://archive.ics.uci.edu/dataset/229/skin+segmentation
        31,     # Covertype (very high instances - 581012, high dimensions - 54) - https://archive.ics.uci.edu/dataset/31/covertype
        602,    # Dry Bean (medium instances - 13611, medium dimensions - 16) - https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
        # 372,    # HTRU2 (medium instances - 17898, low dimensions - 8) - https://archive.ics.uci.edu/dataset/372/htru2
        # 80,     # Optical Recognition of Handwritten Digits (low instances - 5620, high dimensions - 64) - https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
        # 891,    # CDC Diabetes Health Indicators (high instances - 253680, medium dimensions - 21) - https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
        78,     # Page Blocks Classification (medium instances - 5473, medium dimensions - 10) - https://archive.ics.uci.edu/dataset/78/page+blocks+classification
        545,    # Rice (Cammeo and Osmancik) (medium instances - 3810, medium dimensions - 7) - https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
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
        now = datetime.now().strftime('%H:%M:%S')
        print(f"\n({now}) Classification of dataset {metadata.name} has started...")
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

        now = datetime.now().strftime('%H:%M:%S')
        print(f"({now}) Classification of dataset {metadata.name} has ended.")


    if option == "--save":
        file_name = save_results_csv(results)
        print(f"\nGenerated csv file with results: {file_name}")
    elif option == "--print":
        print_results(results=results)
