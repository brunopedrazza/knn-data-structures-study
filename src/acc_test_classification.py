import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from kd_tree import KdTree
from knn import KNN

if __name__ == "__main__":
    dataset = datasets.load_digits()


    # dataset = datasets.fetch_covtype()
    # data, target = datasets.make_classification(n_samples=10000, n_features=50, n_classes=10, n_informative=10, shuffle=True, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.1, random_state=42)

    kd_tree = KdTree.construct(X_train, y_train)
    y_pred = kd_tree.predict(X_test)
    # l = kd_tree.dfs()
    
    # for e in l:
    #     print(e[1])
    # my implemenation
    # knn = KNN(k=9)
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test, n_chunks=256)

