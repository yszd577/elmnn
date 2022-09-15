import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from models.metric_learn.utils import bin_DDAE, bin_DDAE_new
# sys.path.append('..')

from models.metric_learn_new import LMNN


class LmnnIter():
    def __init__(self, init='auto', k=3, min_iter=50, max_iter=100000,
                 learn_rate=1e-7, regularization=0.5, convergence_tol=0.001,
                 verbose=False, preprocessor=None,
                 n_components=None, random_state=None, mode=1, bin_num=10):
        self.init = init
        self.k = k
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.regularization = regularization
        self.convergence_tol = convergence_tol
        self.verbose = verbose
        self.n_components = n_components
        self.random_state = random_state
        self.mode = mode
        self.bin_num = bin_num

    def transform(self, X):
        assert (len(self.tr_) == 2)
        return X.dot(self.tr_[1].T)

    def fit(self, X, y):
        # preprocessing
        if self.mode == 1:
            db = bin_DDAE(X, y)
        elif self.mode == 2:
            db = bin_DDAE_new(X, y, self.bin_num)
        # db = bin_DDAE(X, y)
        tr = []
        lmnn_clf = LMNN()
        n_components = X.shape[1]
        tr.append(np.identity(n_components))
        for db_ in db:
            X_, y_ = db_[:, :-1], db_[:, -1]
            X_ = X_.dot(tr[-1].T)
            lmnn_clf.fit(X_, y_)
            tr.append(np.matmul(lmnn_clf.components_, tr[-1]))
        self.db_ = db
        self.tr_ = tr

    def neigh_predict(self, X):
        knn = KNeighborsClassifier(n_neighbors=3)
        db = self.db_
        tr = self.tr_
        prediction_temp = []
        for block, tr_ in list(zip(db, tr[1:])):
            X1 = block[:, :-1]
            y1 = block[:, -1].astype(int)
            knn.fit(np.matmul(X1, tr_.T), y1)
            inds = knn.kneighbors(np.matmul(X, tr_.T), return_distance=False)
            labels = y1[inds]
            prediction_temp.append(np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=labels))
        prediction = np.sum(prediction_temp, axis=0)
        y_pred = np.argmax(prediction, axis=1)
        return y_pred
