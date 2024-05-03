from time import perf_counter

from numpy import ndarray
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

def divide_chunks(l, n):
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
        

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        tic = perf_counter()
        result = func(*args, **kwargs)
        toc = perf_counter()
        print(f"Function {func.__qualname__} took {toc - tic:.4f} seconds to execute")
        return result
    return wrapper

def euclidean_distance(p1, p2):
    if len(p1.shape) == 1:
        return np.sqrt(np.sum(np.square(p1 - p2)))
    return distance.cdist(p1, p2, 'euclidean')

class KNeighborsClassifierW(KNeighborsClassifier):

    @measure_execution_time
    def predict(self, X) -> ndarray:
        return super().predict(X)
    
    @measure_execution_time
    def fit(self, X, y) -> ndarray:
        return super().fit(X, y)
