import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from models.ddae import DDAE
from probebins import get_bins
from models.metric_learn.lmnn1 import LMNN as lmnn1
from models.metric_learn.utils import benchmark
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from models.rp import RP

# df = pd.read_csv('datasets/benchmark/kr-vs-k-zero_vs_eight.csv')
# cat_features = []
# for col in df.columns[:-1]:
#     if str(df[col].dtype) == 'object':
#         cat_features.append(col)
# data_dummies = pd.get_dummies(df[cat_features])
# df.drop(columns=cat_features, inplace=True)
# df_new = pd.concat([data_dummies, df], axis=1)
# le = LabelEncoder()
# label = df.columns[-1]
# df_new[label] = le.fit_transform(df[label])
# print(np.bincount(df_new[label]))

path = 'datasets/benchmark/kc1.csv'
df = pd.read_csv(path)
X, y = df.values[:, :-1].astype(float), df.values[:, -1].astype(int)
print(np.bincount(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf6 = SelfPacedEnsembleClassifier(base_estimator=DecisionTreeClassifier())
clf6.fit(X_train, y_train)
y_pred = clf6.predict(X_test)
benchmark(y_test, y_pred)
print('*' * 10)





