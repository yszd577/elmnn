import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from probebins import get_bins
from models.metric_learn.lmnn1 import LMNN as lmnn1
from models.metric_learn.lmnn2 import LmnnIter as lmnn2
from models.ddae import DDAE
from models.rp import RP
from models.metric_learn_new.lmnn import LMNN
from models.metric_learn.utils import benchmark
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
import os
import sys
import traceback

root_path = 'datasets'
data_path = os.listdir(root_path)
data_path.sort()
for data_path_ in data_path:
    data_path_inner = os.path.join(root_path, data_path_)
    for file in os.listdir(data_path_inner):
        path = os.path.join(data_path_inner, file)
        result_path = os.path.join('result-final-final', data_path_, file[:-4] + '.log')
        if os.path.isfile(result_path):
            continue

        result_print = open(result_path, 'w')
        sys.stdout = result_print

        bins = []
        try:
            for i in range(3):
                bins.append(get_bins(path, random_state=44))
        except:
            traceback.print_exc()
        else:
            bins.sort()
            bin = int(bins[1])


            print(file[:-4])
            print("bin:", bin)

            df = pd.read_csv(path)
            X, y = df.values[:, :-1].astype(float), df.values[:, -1].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=44)
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            print("lmnn1:")
            clf1 = lmnn1(k=5, bin_num=bin, mode=2)
            clf1.fit(X_train, y_train)
            y_pred = clf1.neigh_predict(X_test)
            benchmark(y_test, y_pred)
            print('*' * 10)

            print("lmnn2:")
            clf2 = lmnn2(k=5, bin_num=bin, mode=2)
            clf2.fit(X_train, y_train)
            y_pred = clf2.neigh_predict(X_test)
            benchmark(y_test, y_pred)
            print('*' * 10)

            print("lmnn:")
            clf3 = LMNN(k=5)
            clf3.fit(X_train, y_train)
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(clf3.transform(X_train), y_train)
            y_pred = knn.predict(clf3.transform(X_test))
            benchmark(y_test, y_pred)
            print('*' * 10)

            print("self-paced:")
            clf4 = SelfPacedEnsembleClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5))
            clf4.fit(X_train, y_train)
            y_pred = clf4.predict(X_test)
            benchmark(y_test, y_pred)
            print('*' * 10)

            print("DDAE:")
            y_maj = y_train[y_train == 0]
            y_min = y_train[y_train == 1]
            ratio = len(y_maj) / len(y_min)
            clf5 = DDAE(x=ratio, k=5)
            clf5.fit(X_train, y_train)
            y_pred = clf5.predict(X_test)
            benchmark(y_test, y_pred)
            print('*' * 10)

            print("RP:")
            clf6 = RP()
            clf6.fit(X_train, y_train)
            y_pred = clf6.predict(X_test)
            benchmark(y_test, y_pred)
            print('*' * 10)

            print("xgb:")
            clf7 = XGBClassifier()
            clf7.fit(X_train, y_train)
            y_pred = clf7.predict(X_test)
            benchmark(y_test, y_pred)
            result_print.close()


