import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.metric_learn.utils import benchmark, bin_DDAE
from xgboost.sklearn import XGBClassifier
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")
xgb.set_config(verbosity=0)

root_path = 'datasets'
data_path = os.listdir(root_path)
data_path.sort()
for data_path_ in data_path:
    data_path_inner = os.path.join(root_path, data_path_)
    for file in os.listdir(data_path_inner):
        path = os.path.join(data_path_inner, file)
        result_path = os.path.join('result-xg', data_path_, file[:-4] + '.log')
        if os.path.isfile(result_path):
            continue

        result_print = open(result_path, 'w')
        sys.stdout = result_print

        print(file[:-4])

        df = pd.read_csv(path)
        X, y = df.values[:, :-1].astype(float), df.values[:, -1].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=44)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        db = bin_DDAE(X_train, y_train)
        dtrains = []
        for db_ in db:
            dtrains.append(xgb.DMatrix(db_[:, :-1], label=db_[:, -1]))
        dtest = xgb.DMatrix(X_test)

        print("xgb-1:")
        param = {'max_depth': 4, 'eta': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
        num_round = 10
        bst = xgb.train(param, dtrains[0], num_round)
        y_pred = bst.predict(dtest)
        y_pred = (y_pred >= 0.5) * 1
        benchmark(y_test, y_pred)
        print('*' * 10)

        print("xgb-final:")
        param_ = {'max_depth': 4, 'eta': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'process_type': 'update',
                  'updater': 'refresh'}
        evals = [(dtrains[0], 'Train0')]
        for i in range(1, len(dtrains)):
            evals.append((dtrains[i], f'Train{i}'))
            bst = xgb.train(param_, dtrains[i], num_round, evals=evals, xgb_model=bst, verbose_eval=False)
            y_pred = bst.predict(dtest)
            y_pred = (y_pred >= 0.5) * 1
            if i == len(dtrains) - 1:
                benchmark(y_test, y_pred)
        print('*' * 10)

        print("spe + dt:")
        spe = SelfPacedEnsembleClassifier(base_estimator=DecisionTreeClassifier())
        spe.fit(X_train, y_train)
        y_pred = spe.predict(X_test)
        benchmark(y_test, y_pred)
        print('*' * 10)

        print("spe + xg:")
        spe = SelfPacedEnsembleClassifier(base_estimator=XGBClassifier())
        spe.fit(X_train, y_train)
        y_pred = spe.predict(X_test)
        benchmark(y_test, y_pred)
        print('*' * 10)

        
        result_print.close()