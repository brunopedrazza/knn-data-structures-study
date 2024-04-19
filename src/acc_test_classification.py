import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from knn import KNN

if __name__ == "__main__":
    dataset = datasets.load_digits()

    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.1, random_state=42)

    # my implemenation
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"acc: {acc:.6f}")

