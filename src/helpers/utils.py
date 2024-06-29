import yaml
import os
import numpy as np
import csv

from collections import defaultdict
from datetime import datetime
from tabulate import tabulate

from helpers.dotdict import dotdict


def find_farthest_point(X, point):
    """ Return the farthest points in the list given a specific point.

    Parameters
    ----------
    X : List[Any]
        List of points to find the farthest one.
    point : Any
        Point to find the farthest one from.
    """
    
    distances = np.linalg.norm(X - point, axis=1)
    farthest_idx = np.argmax(distances)
    return X[farthest_idx]


def get_results_directory():
    """ Method that gets the directory where the results are. """

    return "../assets/results"


def get_results(file_name):
    """ Fetches results data from a result file.

    Parameters
    ----------
    file_name : str
        Name of the file.
    """
    
    result_directory = get_results_directory()
    file_path = os.path.join(result_directory, file_name)

    results = []
    with open(file_path, mode="r", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            results.append(row)
    
    return results


def print_results(file_name=None, results=None):
    if not file_name and not results:
        raise ValueError("File name or actual results are required")
    
    if not results:
        results = get_results(file_name)
    
    db_groups = defaultdict(list)
    for row in results:
        db_groups[row["db_id"]].append(row)
    
    for db_id, db_results in db_groups.items():
        db_metadata = db_results[0]
        db_name, n_samples, n_dimensions = db_metadata["db_name"], db_metadata["n_samples"], db_metadata["n_dimensions"]
        print()
        print(f"id {db_id} ({db_name}), n_samples={n_samples}, n_dimensions={n_dimensions}")
        results = []
        for db_result in db_results:
            accuracy = db_result["accuracy"]
            result = {
                "k": db_result["k"],
                "method": db_result["method"],
                "n_train": db_result["n_train"],
                "n_test": db_result["n_test"],
                "fit_duration_ms": float(db_result["fit_duration_seconds"]) * 1000,
                "predict_duration_seconds": db_result["predict_duration_seconds"],
                "accuracy": f"{float(accuracy)*100:.2f}%",
                "nodes_visited": db_result["nodes_visited"],
                "max_depth": db_result["max_depth"],
            }
            improv = 0
            if result["method"] != "brute_force":
                brute_predict_duration = next((item["predict_duration_seconds"] for item in results if item["method"] == "brute_force" and item["k"] == result["k"]), 0)
                improv = float(result["predict_duration_seconds"]) - float(brute_predict_duration) if float(brute_predict_duration) != 0 else 0
            result["predict_diff_seconds"] = improv
            results.append(result)
        print(tabulate(results, headers="keys", tablefmt="rounded_grid"))
        

def save_results_csv(results):
    """ Save results in a csv file.

    Parameters
    ----------
    results : Any
        Results data.
    """

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


def load_parameters():
    parameters_path = "../parameters.yaml"
    with open(parameters_path, mode="r") as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
    return dotdict(params)
