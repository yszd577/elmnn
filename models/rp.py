import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from models.metric_learn.utils import benchmark


class RP:
    def __init__(self, base_estimators=DecisionTreeClassifier()):
        self.base_estimators = base_estimators

    def fit(self, X, y):
        data = np.hstack((X, y[:, None]))
        db_maj = data[y == 0]
        db_min = data[y == 1]
        k_bins = []

        def generate_bins(X_maj, X_min, temp=0, start=0):
            partition = len(X_maj) // len(X_min)
            for i in range(len(X_min)):
                knn = KNeighborsClassifier()
                knn.fit(X_maj[:, :-1], np.full(len(X_maj), temp))
                idx1 = np.random.choice(len(X_maj), 1)
                idx = knn.kneighbors(X_maj[:, :-1][idx1], n_neighbors=partition, return_distance=False).reshape(
                    partition)
                if i == 0:
                    for j in range(partition):
                        k_bins.append(np.array(X_maj[idx[j]]))
                    X_maj = np.delete(X_maj, idx, axis=0)
                else:
                    for j in range(partition):
                        k_bins[j + start] = np.vstack([k_bins[j + start], X_maj[idx[j]]])
                    X_maj = np.delete(X_maj, idx, axis=0)
            for k in range(start, start + partition):
                k_bins[k] = np.vstack([k_bins[k], X_min])
            if len(X_maj) != 0:
                X_maj, X_min = X_min, X_maj
                temp = not temp
                start += partition
                generate_bins(X_maj, X_min, temp=temp, start=start)

        generate_bins(db_maj, db_min)
        estimators = []
        for a in range(len(k_bins)):
            dt = DecisionTreeClassifier()
            dt.fit(k_bins[a][:, :-1], k_bins[a][:, -1].astype('int'))
            estimators.append(dt)
        self.estimators = estimators

    def predict(self, X):
        temp1 = np.asarray([est.predict(X) for est in self.estimators]).T
        y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=temp1)
        return y_pred


