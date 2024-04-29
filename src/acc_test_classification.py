import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from kd_tree import KdTree
from knn import KNN
from utils import KNeighborsClassifierW

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

    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"acc: {acc*100:.2f}%")
    
    # sklearn
    knn2 = KNeighborsClassifierW(n_neighbors=5, algorithm="kd_tree")
    knn2.fit(X_train, y_train)
    y_pred2 = knn2.predict(X_test)

    acc2 = metrics.accuracy_score(y_test, y_pred2)
    print(f"acc: {acc2*100:.2f}%")

