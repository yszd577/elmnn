import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.metric_learn.lmnn1 import LMNN as lmnn1
from models.metric_learn.utils import benchmark
from sklearn.neighbors import KNeighborsClassifier
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier


df = pd.read_csv('datasets/real/scene.csv')
X, y = df.values[:, :-1].astype(float), df.values[:, -1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=44)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print_log = open('binsearch-scene.log', 'w')
sys.stdout = print_log

bin_num = [*range(10, 21, 1)]

# self-paced
spe = SelfPacedEnsembleClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5))
spe.fit(X_train, y_train)
y_pred = spe.predict(X_test)
benchmark(y_test, y_pred)
print()

# elmnn
for num in bin_num:
    print(num)
    clf = lmnn1(k=5, mode=2, bin_num=num)
    clf.fit(X_train, y_train)
    y_pred = clf.neigh_predict(X_test)
    benchmark(y_test, y_pred)
    print('*' * 10)
