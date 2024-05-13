import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from knn import KNN
from utils import KNeighborsClassifierW

if __name__ == "__main__":
    # dataset = datasets.fetch_covtype()
    # data = dataset.data
    # target = dataset.target
    k = 5

    data, target = datasets.make_classification(
        n_samples=30000, 
        n_features=100, 
        n_informative=100, 
        n_redundant=0, 
        n_repeated=0, 
        n_classes=10, 
        shuffle=True, 
        random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    print("Len X_train " + str(len(X_train)))
    print("Len X_test " + str(len(X_test)))

    # my kd tree leaf
    knn = KNN(k=k, method="kd_tree", leaf_size=100)
    knn.fit(X_train, y_train)
    y_pred, distance_count = knn.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"pred my kd tree acc: {acc*100:.2f}%")
    print(f"distance count: {distance_count}")

    # my brute force
    # knn = KNN(k=k, method="brute_force")
    # knn.fit(X_train, y_train)
    # y_pred, distance_count = knn.predict(X_test)

    # acc = metrics.accuracy_score(y_test, y_pred)
    # print(f"pred my brute acc: {acc*100:.2f}%")
    # print(f"distance count: {distance_count}")

    # sklearn kd tree
    knn2 = KNeighborsClassifierW(n_neighbors=k, algorithm="kd_tree")
    knn2.fit(X_train, y_train)
    y_pred2 = knn2.predict(X_test)

    acc2 = metrics.accuracy_score(y_test, y_pred2)
    print(f"pred kd_tree sklearn acc: {acc2*100:.2f}%")

    # sklearn ball tree
    knn2 = KNeighborsClassifierW(n_neighbors=k, algorithm="ball_tree")
    knn2.fit(X_train, y_train)
    y_pred2 = knn2.predict(X_test)

    acc2 = metrics.accuracy_score(y_test, y_pred2)
    print(f"pred ball_tree sklearn acc: {acc2*100:.2f}%")

