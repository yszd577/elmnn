import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from models.metric_learn.lmnn2 import LmnnIter as lmnn2
from sys import argv


def get_bins(input=str, random_state=int):
    K = 3
    S = K
    # print(argv)
    if len(argv) > 1:
        input = argv[1]
        K = int(argv[2])
        # S = int(argv[3])
    S = 1
    # print(input, K, S)


    def updatevsmaj(input_all):
        # input_maj = maj[0][input]
        Xn = X_train[input_all]
        yn = y_train[input_all]
        clf = lmnn2(mode=2, bin_num=1, k=K, n_components=3)
        clf.fit(Xn, yn)
        return clf


    def update(count):
        input = idx[0: count]
        input_all = np.union1d(major[0][input], minor[0])
        return updatevsmaj(input_all)


    def filter(clf, valid):
        # input_all = np.union1d(input, minor[0])
        # Xn = X_train[input_all]
        # yn = y_train[input_all]
        # lmnn_clf = lmnn2(mode=2, bin_num=1, k=K, n_components=3)
        # lmnn_clf.fit(Xn, yn)
        valid_all = np.union1d(valid, minor[0])
        Xt = X_train[valid_all]
        yt = y_train[valid_all]
        yp = clf.neigh_predict(Xt)
        # benchmark(yt, yp)
        y_merg = yt * yp
        idx_bnd_maj = []
        for i in range(0, len(y_merg)):
            if y_merg[i] == 0 and yp[i] == 1 and yt[i] == 0:
                idx_bnd_maj.append(valid_all[i])
        return idx_bnd_maj


    def avgdistvsmaj(clf, input_maj):
        dists_maj, dists_min = [], []
        knn.fit(clf.transform(pd.DataFrame(X_train).loc[input_maj].values),
                pd.DataFrame(y_train).loc[input_maj].values.ravel())
        dist1, ind1 = knn.kneighbors()
        dists_maj.append(np.mean(dist1[:, :3]))

        knn.fit(clf.transform(pd.DataFrame(X_train).loc[minor[0]].values),
                pd.DataFrame(y_train).loc[minor[0]].values.ravel())
        dist2, ind2 = knn.kneighbors()
        dists_min.append(np.mean(dist2[:, :3]))
        return dists_maj, dists_min


    def probing(clf, input, val=False):
        bf = int(1.2 * len(major[0]) / len(minor[0]))
        Xd = clf.transform(X_train)
        left = pd.DataFrame(Xd).loc[input]
        ridx = [m for m in major[0] if m not in input]
        right = pd.DataFrame(Xd).loc[ridx]
        knn = KNeighborsClassifier(n_neighbors=int(bf))
        knn.fit(right, np.zeros(len(ridx)))
        for p in range(int(bf), 1, -1):
            output = np.array(input).tolist()
            dists, index = knn.kneighbors(left)
            # dists, index = knn.kneighbors(pd.DataFrame(Xd).loc[output])
            cnt = int(len(major[0]) / p)
            k = 0
            step = S
            found = False
            while True:
                for i in range(0, len(input)):
                    # for i in range(len(output), 0, -1):
                    e = k + step
                    if e > bf: e = bf
                    for j in range(k, e):
                        if ridx[index[i][j]] not in output:
                            output.append(ridx[index[i][j]])
                            if len(output) >= cnt:
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
                k = k + step
                if k >= int(bf):
                    break
            dists_maj, dists_min = avgdistvsmaj(clf, output)
            # print("***", p, step, k, int(cnt), int(bf), len(output), len(left), len(right), dists_maj, dists_min)
            if dists_maj < dists_min:
                return p + 1
        return 1


    def heuristic(wrong, minor, major):
        # old = len(wrong)
        # xs = set(wrong)
        inner = wrong[0:len(minor[0])]
        outer = []
        for i in range(0, len(major[0])):
            if major[0][i] not in inner: outer.append(major[0][i])
        clf = updatevsmaj(np.union1d(inner, minor[0]).astype(int))
        wrong = filter(clf, outer)
        # dfrt = xs.intersection(wrong)
        # diff = np.linalg.norm(np.inner(lmnn_clf.tr_[1].T, clf.tr_[1].T))
        # print('wrong:', old, len(wrong), diff, 'metric *******************************************')
        # print(len(wrong))
        return wrong


    df = pd.read_csv(input)
    X, y = df.values[:, :-1].astype(float), df.values[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    major = np.where(y_train <= 0.5)
    minor = np.where(np.array(y_train) > 0.5)
    idx = np.arange(0, len(major[0]), 1)
    random.shuffle(idx)
    more = 0
    extreme = False
    knn = KNeighborsClassifier(n_neighbors=K)
    first_clf = None

    initial_round = int(1.25 * len(major[0]) / len(minor[0]))
    cnt_initial = int(len(major[0]) / initial_round)

    # print(initial_round, cnt_initial)
    lmnn_clf = update(cnt_initial)
    wrong = filter(lmnn_clf, major[0][idx[cnt_initial:]])  # wrong is the real idx in major[0]
    # lmnn_clf = update(np.union1d(wrong, minor[0]))

    if (len(wrong) <= K):
        more = int(len(major[0]) / len(minor[0]))
    else:
        validation = False
        for i in range(0, 2):
            wrong = heuristic(wrong, minor, major)
            if (len(wrong) < K):
                P = 1
                break
        clf = updatevsmaj(np.union1d(wrong, minor[0]).astype(int))
        P = probing(clf, wrong[0:len(minor[0])], validation)
        # print(len(major[0]) / len(minor[0]), P)
        more = P

    # print(more, " =================")
    return more

