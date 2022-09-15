import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from models.metric_learn_new.lmnn import LMNN
from models.metric_learn.utils import DDAE_predict, Dbc


class DDAE():
    def __init__(self, x, k=3, gamma=0.2):
        self.x = x
        self.k = k
        self.gamma = gamma

    def fit(self, X, y):
        # Data Block Construction
        db = Dbc(X, y)
        delta = len(db)

        # Data Space Improvement
        unstable = []
        y_pred = []
        tau = 1
        clf = LMNN(k=self.k)
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn1 = KNeighborsClassifier(n_neighbors=self.k)
        transform = []
        classifier = []
        for block in db:
            unstable1 = []
            y_pred1 = []
            X1, y1 = block[:, :-1].astype(float), block[:, -1].astype(int)
            X1_tr = clf.fit_transform(X1, y1)
            transform.append(clf.components_)
            knn1.fit(X1_tr, y1)
            classifier.append(knn1)
            knn.fit(X1, y1)
            dist, inds = knn.kneighbors(X1)
            for samp in range(len(X1)):
                p_samp, n_samp = 0, 0
                for neigh in range(self.k):
                    if y1[inds[samp][neigh]] == 0:
                        n_samp += 1
                    else:
                        p_samp += 1
                if np.absolute(n_samp - p_samp) <= tau:
                    unstable1.append(block[samp])
            if len(unstable1) != 0:
                unstable1 = np.vstack(unstable1)
                y_pred1 = knn.fit(unstable1[:, :-1])
                unstable.append(unstable1)
                y_pred.append(y_pred1)
        self.transform_ = transform
        self.clf = classifier

        # Adaptive Weight Adjustment
        ratio = self.x
        if delta % 2 == 0:
            Wt = (delta / 2 + 1) / (delta / 2 - 1)
        else:
            Wt = (math.floor(delta / 2) + 1) / (math.floor(delta / 2))
        Delta = 0.1
        Wn = Wt + Delta
        Wd = 1
        default, pos, neg = 0, 0, 0
        co = len(unstable)
        for i in range(co):
            y_true, y_predict = unstable[i][:, -1], y_pred[i]
            c_11 = np.sum((y_true == 1) & (y_pred == 1))
            c_00 = np.sum((y_true == 0) & (y_pred == 0))
            c_10 = np.sum((y_true == 1) & (y_pred == 0))
            c_01 = np.sum((y_true == 0) & (y_pred == 1))
            gain_mat = ratio * (c_11 - c_10) + (c_00 - c_01)
            gain_pos = ratio * (c_11 + c_10) + (-c_00 - c_01)
            gain_neg = ratio * (-c_11 - c_10) + (c_00 + c_01)
            gain_max = max(gain_mat, gain_pos, gain_neg)
            if gain_max == gain_mat:
                default += 1
            elif gain_max == gain_pos:
                pos += 1
            else:
                neg += 1
        if (pos + neg) / delta < self.gamma:
            weight = np.array([Wd, Wd])
        else:
            if neg > pos:
                weight = np.array([Wn, Wd])
            else:
                weight = np.array([Wd, Wn])
        self.weights = weight

        # Ensemble Learning
        # classifier = []
        # for j in db:
        #     X2, y2 = j[:, :-1], j[:, -1]
        #     knn_base = KNeighborsClassifier(n_neighbors=self.k)
        #     knn_base.fit(X2, y2)
        #     classifier.append(knn_base)
        # self.clf = classifier

    def predict(self, X):
        y_pred = DDAE_predict(X, self.transform_, self.clf, self.weights)
        return y_pred





