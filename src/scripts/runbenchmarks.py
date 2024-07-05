import sys

from datetime import datetime
from ucimlrepo.fetch import fetch_ucirepo

from helpers.benchmark import benchmark
from scripts.showresults import print_results
from helpers.utils import load_parameters, save_results_csv


if __name__ == "__main__":

    option_err_message = "Please choose whether you want to save the results or print then, with --save or --print"
    try:
        option = sys.argv[1]
        if option != "--save" and option != "--print":
            raise ValueError()
    except (IndexError, ValueError):
        print(option_err_message)
        exit()

    parameters = load_parameters()
    
    dbs_data = []
    for db_id in parameters.database_ids:
        dataset = fetch_ucirepo(id=db_id)
        X, y = dataset.data.features.values, dataset.data.targets.values
        dbs_data.append((X, y, dataset.metadata))
    
    results = []
    for db_data in dbs_data:
        X, y, metadata = db_data
        now = datetime.now().strftime('%H:%M:%S')
        print(f"\n({now}) Classification of dataset {metadata.name} has started...")
        n_iterations = parameters.n_iterations
        for k in parameters.n_neighbors:
            for method in parameters.methods:
                data = {
                    "db_id": metadata.uci_id,
                    "db_name": metadata.name,
                    "n_samples": len(X),
                    "n_dimensions": len(X[0]),
                    "k": k,
                    "method": method
                }
                result = benchmark(X, y, k, method, parameters.leaf_size, n_iterations=n_iterations)
                data.update(result)
                results.append(data)

        now = datetime.now().strftime('%H:%M:%S')
        print(f"({now}) Classification of dataset {metadata.name} has ended.")


    if option == "--save":
        file_name = save_results_csv(results)
        print(f"\nGenerated csv file with results: {file_name}")
    elif option == "--print":
        print_results(results=results)
