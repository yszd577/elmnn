import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, confusion_matrix, fbeta_score
from itertools import product


def imbalance_train_test_split(X, y, test_size, random_state=None):
    '''Train/Test split that guarantee same class distribution between split datasets.'''
    # np.random.seed(1)
    X_maj = X[y == 0]
    y_maj = y[y == 0]
    X_min = X[y == 1]
    y_min = y[y == 1]
    X_train_maj, X_test_maj, y_train_maj, y_test_maj = train_test_split(
        X_maj, y_maj, test_size=test_size, random_state=random_state)
    X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(
        X_min, y_min, test_size=test_size, random_state=random_state)
    X_train = np.concatenate([X_train_maj, X_train_min])
    X_test = np.concatenate([X_test_maj, X_test_min])
    y_train = np.concatenate([y_train_maj, y_train_min])
    y_test = np.concatenate([y_test_maj, y_test_min])
    return X_train, X_test, y_train, y_test


def load_data(path):
    df = pd.read_csv(f'standard_datasets/imbalance/{path}.csv')
    X, y = df.values[:, :-1], df.values[:, -1]
    if path == 'ecoli' or path == 'yeast':
        X, y = df.values[:, :-1], df.values[:, -1]
        le = LabelEncoder()
        y = le.fit_transform(y)
    X, y = X.astype(float), y.astype(int)
    return X, y


def DDAE_predict(X, transform, classifier, weight=[1, 1]):
    predictions = np.asarray([est.predict(X.dot(trans.T)) for trans, est in zip(transform, classifier)]).T
    predictions = predictions.astype(int)
    maj = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=predictions)
    maj = maj * weight
    y_pred = np.argmax(maj, axis=1)
    return y_pred


def sidePrecicion(y_test, y_pred):
    print(len([y_test for y_test in y_test if y_test == 1]), ":", len(y_test))
    print(len([y_pred for y_pred in y_pred if y_pred == 1]), ":", len(y_pred))
    y_merg = y_pred * y_test
    print(len([y_merg for y_merg in y_merg if y_merg == 1]), ":", len(y_merg))


def bin_DDAE(X, y):
    db = []
    df = np.hstack((X, y[:, None]))
    maj = df[y == 0]
    mio = df[y == 1]
    delta = round(len(maj) / len(mio))
    rng = np.random.default_rng()
    rand = rng.integers(low=0, high=50, size=delta)
    for i_block in range(delta):
        if i_block == delta - 1:
            db.append(np.vstack((maj, mio)))
        else:
            rng1 = np.random.default_rng(rand[i_block])
            idx = rng1.choice(len(maj), len(mio), replace=False)
            db.append(np.vstack((maj[idx], mio)))
            maj = np.delete(maj, idx, axis=0)
    return db


def bin_bpluskc(db, X, y):
    df = np.hstack((X, y[:, None]))
    maj = df[y == 0]
    mio = df[y == 1]
    min_num = len(mio)
    delta = round(len(maj) / len(mio))
    mio = np.repeat(mio, delta, axis=0)
    rng = np.random.default_rng()
    for i in range(delta):
        if i == delta - 1:
            db.append(np.vstack((maj, mio)))
        else:
            idx1 = rng.choice(len(maj), min_num, replace=False)
            idx2 = rng.choice(len(mio), min_num, replace=False)
            db.append(np.vstack((maj[idx1], mio[idx2])))
            maj = np.delete(maj, idx1, axis=0)
            mio = np.delete(mio, idx2, axis=0)


def bin_predict(db: list, k: int, X, clf):
    knn = KNeighborsClassifier(n_neighbors=k)
    min_length = len(db)
    prediction_temp = []
    for block in db:
        X1 = block[:, :-1]
        y1 = block[:, -1].astype(int)
        knn.fit(clf.transform(X1), y1)
        prediction_temp.append(knn.predict(clf.transform(X)))
    prediction = np.vstack(prediction_temp)
    prediction = prediction.T.astype(int)
    maj = np.apply_along_axis(lambda x: np.bincount(x, minlength=min_length), axis=1, arr=prediction)
    y_pred = np.argmax(maj, axis=1)
    return y_pred


def bin_predict_proba(db: list, k: int, X, clf):
    knn = KNeighborsClassifier(n_neighbors=k)
    min_length = len(db)
    prediction_temp = []
    for block in db:
        X1 = block[:, :-1]
        y1 = block[:, -1]
        knn.fit(clf.transform(X1), y1)
        prediction_temp.append(knn.predict_proba(clf.transform(X)))
    proba = sum(prediction_temp) / min_length
    # proba = np.maximum(proba[:, :1], proba[:, 1:])
    return proba


def metric_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn / (tn + fp)
    g_mean = math.sqrt(recall * tnr)
    f_ms = fbeta_score(y_true, y_pred, beta=2)
    return recall, g_mean, f_ms


def draw(X, y, i):
    plt.figure()
    # 蓝 橙
    colors = ['#1F77B4', '#FF7F0E']
    for l, c in zip(np.unique(y), colors):
        plt.scatter(X[y == l, 0], X[y == l, 1], c=c, s=10)
    plt.savefig(f'pic/{i}.png')


# def bin_multi(X, y):
#     unique_labels = np.unique(y)
#     db = []
#     for label in unique_labels:
#         indic1 = (y == label)
#         indic0 = (y != label)
#         X_min = X[indic1]
#         X_maj = X[indic0]
#         y_min = y[indic1]
#         y_maj = y[indic0]
#         y_min[:] = 1
#         y_maj[:] = 0
#         X_merge = np.vstack((X_maj, X_min))
#         y_merge = np.hstack((y_maj, y_min))
#         db_ = []
#         bin_DDAE(db_, X_merge, y_merge)
#         db.append(db_)
#     return db


# def multi_bin(df):
#     y = df[:, -1].astype(int)
#     bins = []
#     db = []
#     labels = np.unique(y)
#     num_labels = np.bincount(y)
#     num_labels = np.delete(num_labels, np.where(num_labels == 0))
#     inds = np.argsort(num_labels)
#     for i in inds:
#         bins.append(df[y == labels[i]])
#     for index in range(len(bins)):
#         db_ = []
#         db_.append(bins[index])
#         num = index + 1
#         for next_bin in bins[index + 1:]:
#             rng = np.random.default_rng()
#             idx = rng.choice(len(next_bin), len(bins[index]), replace=False)
#             db_.append(next_bin[idx])
#             bins[num] = np.delete(bins[num], idx, axis=0)
#             num += 1
#         db_ = np.vstack(db_)
#         db.append(db_)
#     return db


# def average_bin(X, y):
#     df = np.hstack((X, y[:, None]))
#     labels = np.unique(y)
#     num_labels = np.bincount(y)
#     num_labels = np.delete(num_labels, np.where(num_labels == 0))
#     num = np.amin(num_labels)
#     k = round(np.amax(num_labels) / num)
#     db = []
#     for label in labels:
#         db.append(df[y == label])
#     db1 = []
#     rng = np.random.default_rng()
#     for db_ in db:
#         if len(db_) == num:
#             db1.append(db_)
#         else:
#             delta = round(len(db_) / num)
#             db_sub = []
#             for i in range(delta):
#                 if i == delta - 1:
#                     db_sub.append(db_)
#                 else:
#                     idx = rng.choice(len(db_), num, replace=False)
#                     db_sub.append(db_[idx])
#                     db_ = np.delete(db_, idx, axis=0)
#             db1.append(db_sub)
#     db_final = []
#     for i in range(k):
#         db_final_ = []
#         for db1_ in db1:
#             if len(db1_) == num:
#                 db_final_.append(db1_)
#             else:
#                 db_final_.append(db1_[((i+1) % len(db1_)) - 1])
#         db_final_ = np.vstack(db_final_)
#         db_final.append(db_final_)
#     return db_final


# def random_bin(X, y):
#     df = np.hstack((X, y[:, None]))
#     labels = np.unique(y)
#     num_labels = np.bincount(y)
#     num_labels = np.delete(num_labels, np.where(num_labels == 0))
#     length = np.amin(num_labels)
#     k = round(np.amax(num_labels) / length)
#     db = []
#     for label in labels:
#         db.append(df[y == label])
#     db_final = []
#     for i in range(k):
#         db_final_ = []
#         for db_ in db:
#             rng = np.random.default_rng()
#             idx = rng.choice(len(db_), length, replace=False)
#             db_final_.append(db_[idx])
#         db_final_ = np.vstack(db_final_)
#         db_final.append(db_final_)
#     return db_final


# def multi_predict(db: list, k: int, labels: list, X, clf):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     prediction_temp = []
#     min_length = len(db)
#     for block in db:
#         X1 = block[:, :-1]
#         y1 = block[:, -1].astype(int)
#         knn.fit(clf.transform(X1), y1)
#         prediction_temp.append(knn.predict(clf.transform(X)))
#     prediction = np.vstack(prediction_temp)
#     prediction = prediction.T.astype(int)
#     maj = np.apply_along_axis(lambda x: np.bincount(x, minlength=min_length), axis=1, arr=prediction)
#     y_pred_index = np.argmax(maj, axis=1)


def all_bin(X, y):
    df = np.hstack((X, y[:, None]))
    labels = np.unique(y)
    num_labels = np.bincount(y)
    num_labels = np.delete(num_labels, np.where(num_labels == 0))
    label_min = np.amin(num_labels)
    db = []
    for label in labels:
        db.append(df[y == label])
    db1 = []
    num_bins = []
    for db_ in db:
        db1_ = []
        delta = round(len(db_) / label_min)
        num_bins.append(list(range(delta)))
        for i in range(delta):
            if i == delta - 1:
                db1_.append(db_)
            else:
                rng = np.random.default_rng()
                idx = rng.choice(len(db_), label_min, replace=False)
                db1_.append(db_[idx])
                db_ = np.delete(db_, idx, axis=0)
        db1.append(db1_)
    index = list(product(*num_bins))
    db_final = []
    for idx in index:
        db_final_ = []
        for a1, a2 in enumerate(idx):
            db_final_.append(db1[a1][a2])
        db_final_ = np.vstack(db_final_)
        db_final.append(db_final_)
    return db_final


def neigh_predict(db: list, k: int, label_num: int, X, clf):
    knn = KNeighborsClassifier(n_neighbors=k)
    prediction_temp = np.zeros((len(X), label_num + 1))
    for db_ in db:
        X1, y1 = db_[:, :-1].astype(float), db_[:, -1].astype(int)
        knn.fit(clf.transform(X1), y1)
        inds = knn.kneighbors(clf.transform(X), return_distance=False)
        label = y1[inds]
        for row in range(len(label)):
            for col in range(k):
                prediction_temp[row, label[row, col]] += 1
    prediction = np.argmax(prediction_temp, axis=1)
    return prediction


def print_label_num(y):
    label = np.bincount(y)
    print(label[label != 0])


def new_bin(X, y):
    df = np.hstack((X, y[:, None]))
    labels = np.unique(y)
    num_labels = np.bincount(y)
    num_labels = np.delete(num_labels, np.where(num_labels == 0))
    bin_len = round(np.amax(num_labels) / 10)
    db = []
    for label in labels:
        db.append(df[y == label])
    db1 = []
    num_bins = []
    for db_ in db:
        if len(db_) < bin_len:
            db1.append([db_])
            num_bins.append([0])
        else:
            delta = int(round(len(db_) / bin_len))
            db1_ = []
            num_bins.append(list(range(delta)))
            for i in range(delta):
                if i == delta - 1:
                    db1_.append(db_)
                else:
                    rng = np.random.default_rng()
                    idx = rng.choice(len(db_), bin_len, replace=False)
                    db1_.append(db_[idx])
                    db_ = np.delete(db_, idx, axis=0)
            db1.append(db1_)
    index = list(product(*num_bins))
    db_final = []
    for idx in index:
        db_final_ = []
        for a1, a2 in enumerate(idx):
            db_final_.append(db1[a1][a2])
        db_final_ = np.vstack(db_final_)
        db_final.append(db_final_)
    return db_final


def multi_prediction_print(y_score, y_true):
    label = np.unique(y_true)
    for y_ in label:
        idx = y_true == y_
        print(len([y_score_ for y_score_ in y_score[idx] if y_score_ == y_]), ';', np.sum(idx))


def neigh_predict_remove(db: list, k: int, label_num: int, X, clf):
    knn = KNeighborsClassifier(n_neighbors=k)
    neigh = []
    prediction_temp = np.zeros((len(X), label_num + 1))
    num = 0
    for db_ in db:
        if num == 0:
            X1, y1 = db_[:, :-1].astype(float), db_[:, -1].astype(int)
            knn.fit(clf.transform(X1), y1)
            inds = knn.kneighbors(clf.transform(X), return_distance=False)
            for row in range(len(inds)):
                X_neigh = []
                for col in range(k):
                    X_neigh_ = db_[inds[row][col]]
                    X_neigh.append(X_neigh_)
                neigh.append(np.vstack((X_neigh)))
        else:
            X1, y1 = db_[:, :-1].astype(float), db_[:, -1].astype(int)
            knn.fit(clf.transform(X1), y1)
            inds = knn.kneighbors(clf.transform(X), return_distance=False)
            for row in range(len(inds)):
                X_neigh = []
                for col in range(k):
                    X_neigh_ = db_[inds[row][col]]
                    X_neigh.append(X_neigh_)
                neigh[row] = np.vstack((neigh[row], np.vstack(X_neigh)))
        num += 1
    prediction = []
    for neigh_ in neigh:
        neigh_ = np.unique(neigh_, axis=0)
        prediction_ = np.argmax(np.bincount(neigh_[:, -1].astype(int)))
        prediction.append(prediction_)
    prediction = np.array(prediction)
    return prediction
