# K-Nearest Neighbors Study

A python application that implements a variety of tree-based data structures to improve the [K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) classification time.

## Pre-requisites
In order for the application to work properly, you need to have [Python](https://www.python.org/downloads/) installed.

## Installation

To install locally, you must follow these steps:
1. Clone the repository locally `$git clone <repo>`
2. cd into the repository directory `$cd knn-data-structures-study`
3. Create e python virtual environment `$python -m venv venv`
4. Activate the virtual environment `$source venv/bin/activate`
5. Install all requirements with `$pip install -r requirements.txt`

## Running the benchmark
To run the application locally, cd into the `/src` directory and run `$python -m scripts.runbenchmarks --save` to save the results on a csv file or `$python -m scripts.runbenchmarks --print` if you only want to see the results tabulated into your terminal.

There is a pre-populated parameters file named `/parameters.yaml` that contains all the parameters needed to run the benchmark:

1. `n_neighbors` defines how many neighbors do you want to get (default: 1, 3 and 5)
2. `methods` defines which methods do you want to use (default: brute_force, kd_tree, ball_tree, ball*_tree and vp_tree)
3. `database_ids` defines the databases ids from [UC Irvine dataset repo](https://archive.ics.uci.edu/datasets) that you want to use to run the benchmark.
4. `leaf_size` defines how many points are going to be stored in leaf nodes, only applies for tree-based methods.
5. `n_iterations` defines how many times to repeat the benchmarks in order to refine the results.

The database ids are already filled with a variety of datasets, each one with a characteristic that benefits from a specific method. Feel free to remove or add more if needed.

When finished, a table of results will be shown on your terminal, or a csv file is going to be saved on `/assets/results` with the date and time of the run. It depends wheter you ran the benchmark with `--save` or `--print`.

### Showing results
If you choose to save the results, you can always come back and run `$python -m scripts.showresults <file-name>` to print the results on the terminal based on the results from a previously generated results file.

With `--save` the results will only be printed once.


#### Each line of the file will contain a separate benchmark with this information:

- `db_id`: Unique id of the dataset on UC Invine repository.
- `db_name`: Human-friendly name of the dataset.
- `n_samples`: Number of samples of the dataset.
- `n_dimensions`: Number of dimensions of the dataset. (n features)
- `k`: Number of neighbors considered to classify test points.
- `method`: Method used to improve (or not) the algorithm performance. 
- `n_train`: Number of samples used for training.
- `n_test`: Number of samples used for testing.
- `fit_duration_seconds`: How many seconds in average it took to fit all training points.
- `predict_duration_seconds`: How many seconds in average it took to classify all test points.
- `accuracy`: Accuracy of the results.
- `nodes_visited`: How many nodes were visited during classification time.
- `max_depth`: Depth of the tree, indicates if the tree is well balanced or not.

## Folder Structure

#### `/assets/datasets` 
All the datasets are stored here in a csv file to avoid going to the external repository of datasets every time the benchmark runs.

#### `/assets/results` 
The results of each benchmark is going to be stored here as a csv file.

**NOTE: The /assets directory will be created on demand, as soon as you run the branchmark for the first time. Hence, it's not included in the repository initially.**

#### `/src/helpers` 
Here we have all the helper methods used by the application, including:

- `heap.py`: Implementation of a max heap to help to always maintain the k-nearest neighbor points. The most distant one (the first to be removed if a closer point is found) is always at the first position of the heap.
- `utils.py`: Some shared, util methods used by the application. Ex: Method to calculate Euclidian distance between points.

#### `/src/scripts`
Entry point files to run benchmarks, plot results and show results.

- `runbenchmarks.py`: Gets the parameters from the configuration file, run the benchmarks using KNN class and store the results on a vsc file or print them in the terminal.
- `showresults.py`: Receives a file name containing results, fetches the results of this file and print the results in a human-friendly way.
- `runfitcomparison.py`: Runs a series of tests to compare the time for fitting between the available methods, increasing the number of samples. After that, it plots the results.
- `runpredictcomparison.py`: Receives a file name containing results, fetches the results of this file and plot them in a bar graph.

#### `/src/knn.py`
The cortex of the application. It has the class that manages all the things needed to run the algorithm with the chosen parameters. It is responsible for initializing the trees, doing distance calculations and for classifying the points.

#### `/src/trees` 
Here we have the implementation of all the tree-based classes to use for classification. Including KdTree, BallTree and VpTree.

#### `/src/trees/nodes` 
The implementation of tree-based nodes are implemented here. Contains the methods for constructing the trees.

#### `/src/ucimlrepo` 
Contains the implementation responsible to get the datasets from UCI repository and save them locally.


## Examples of usage

### Classification in a 2d dataset

```python
from knn import KNN

points = [
    [2,4],[1,3],[2,5],[3,2],[2,1],
    [5,6],[4,5],[3,6],[6,6],[5,4]
]
targets = [
    "blue", "blue", "blue", "blue", "blue",
    "red", "red", "red", "red", "red"
]

new_points = [[1,1],[2,3],[4,4]]

knn = KNN(k=1, method="kd_tree", leaf_size=2)
knn.fit(points, targets)
new_classes, _ = knn.predict(new_points)

print(new_classes)

>>> ['blue' 'blue' 'red']
```
