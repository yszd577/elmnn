import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

for file in os.listdir('data/'):
    print(file[:-4])
    input = os.path.join('data', file)
    df = pd.read_csv(input)
    X, y = df.values[:, :-1].astype(float), df.values[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=44)
    print(np.shape(X), len(np.where(np.array(y) <= 0.5)[0]) / len(np.where(np.array(y) > 0.5)[0]))
