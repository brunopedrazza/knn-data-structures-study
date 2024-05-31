import csv
import os
import sys

from collections import defaultdict
from tabulate import tabulate
from helpers.utils import get_results_directory


def print_results(file_name=None, results=None):
    if not file_name and not results:
        raise ValueError("File name or actual results are required")
    
    if not results:
        result_directory = get_results_directory()
        file_path = os.path.join(result_directory, file_name)

        results = []
        with open(file_path, mode="r", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            for row in reader:
                results.append(row)
    
    db_groups = defaultdict(list)
    for row in results:
        db_groups[row["db_id"]].append(row)
    
    for db_id, db_results in db_groups.items():
        db_metadata = db_results[0]
        print()
        print(f"id {db_id} ({db_metadata["db_name"]}), n_samples={db_metadata["n_samples"]}, n_dimensions={db_metadata["n_dimensions"]}")
        results = []
        for db_result in db_results:
            result = {
                "k": db_result["k"],
                "method": db_result["method"],
                "n_train": db_result["n_train"],
                "n_test": db_result["n_test"],
                "fit_duration_ms": float(db_result["fit_duration_seconds"]) * 1000,
                "predict_duration_seconds": db_result["predict_duration_seconds"],
                "accuracy": f"{float(db_result["accuracy"])*100:.2f}%",
                "distance_count": db_result["distance_count"],
            }
            improv = 0
            if result["method"] != "brute_force":
                brute_predict_duration = next((item["predict_duration_seconds"] for item in results if item["method"] == "brute_force" and item["k"] == result["k"]), 0)
                improv = float(result["predict_duration_seconds"]) - float(brute_predict_duration) if float(brute_predict_duration) != 0 else 0
            result["predict_diff_seconds"] = improv
            results.append(result)
        print(tabulate(results, headers="keys", tablefmt="rounded_grid"))


if __name__ == "__main__":
    try:
        file_name = sys.argv[1]
        print_results(file_name=file_name)
    except IndexError:
        print("\nPlease provide the file name to show results.")
