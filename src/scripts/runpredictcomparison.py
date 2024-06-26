import sys
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from operator import itemgetter

from helpers.utils import get_results


def run_comparison(results):
    db_ids = ["229", "53"]
    results = [d for d in results if d["k"] == "1" and d["db_id"] in db_ids and d["method"] != "brute_force"]
    results.sort(key=lambda item: int(itemgetter("n_samples")(item)))
    db_groups = defaultdict(list)
    for row in results:
        db_groups[row["n_samples"]].append(row)
    
    ro = {}
    for n_samples, result in db_groups.items():
        ri = {}
        for r in result:
            ri[r["method"]] = r["predict_duration_seconds"]
        ro[f"{n_samples};{result[0]["n_dimensions"]}"] = ri

    num_groups = len(ro)
    index = np.arange(num_groups)
    bar_width = 0.2
    _, ax = plt.subplots(figsize=(14, 8))
    vibrant_colors = ['orange', 'dodgerblue', 'limegreen', 'hotpink']

    labels = list(ro[list(ro.keys())[0]].keys())
    for i, label in enumerate(labels):
        values = [float(entry[label]) for entry in ro.values()]
        ax.bar(index + i * bar_width, values, bar_width, label=label, color=vibrant_colors[i])

    ax.set_xlabel('n_samples')
    ax.set_ylabel('Time (s, log scale)')
    ax.set_title('Performance Comparison')
    ax.set_xticks(index + bar_width * (len(labels) - 1) / 2)
    ax.set_xticklabels(ro.keys())
    # ax.set_yscale('log')  # Keeping the y-axis logarithmic for better visibility
    ax.legend()

    plt.show()


if __name__ == "__main__":
    try:
        file_name = sys.argv[1]
        results = get_results(file_name)
        run_comparison(results)
    except IndexError:
        print("\nPlease provide the file name to get results.")

