import math
import time
from scipy.stats import mode
from operator import itemgetter
from .utils import bin_DDAE, bin_DDAE_new
import numpy as np
from collections import Counter
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from .util import _initialize_components, _check_n_components


class LMNN():
    def __init__(self, init='auto', db=[], k=3, min_iter=50, max_iter=100000,
                 learn_rate=1e-7, regularization=0.5, convergence_tol=0.001,
                 verbose=False, n_components=None, random_state=None, mode=1, bin_num=10):
        self.init = init
        self.k = k
        self.db = db
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

    def fit(self, X, y):
        # preprocessing
        if len(self.db) == 0:
            if self.mode == 1:
                db = bin_DDAE(X, y)
            elif self.mode == 2:
                db = bin_DDAE_new(X, y, self.bin_num)
            else:
                raise ValueError("wrong mode")
        else:
            db = self.db
        feature = X.shape[1]
        # train
        k = self.k
        reg = self.regularization
        learn_rate = self.learn_rate
        grads = np.zeros((feature, feature))
        objectives = 0
        neighbors = []
        dfGs = []
        delta = len(db)

        for block in range(delta):
            X, y = db[block][:, :-1].astype(float), db[block][:, -1].astype(int)
            num_pts, d = X.shape
            output_dim = _check_n_components(d, self.n_components)
            unique_labels, label_inds = np.unique(y, return_inverse=True)
            if len(label_inds) != num_pts:
                raise ValueError('Must have one label per point.')
            self.labels_ = np.arange(len(unique_labels))

            self.components_ = _initialize_components(output_dim, X, y, self.init,
                                                      self.verbose,
                                                      random_state=self.random_state)
            required_k = np.bincount(label_inds).min()
            if self.k > required_k:
                raise ValueError('not enough class labels for specified k'
                                 ' (smallest class has %d)' % required_k)

            target_neighbors = self._select_targets(X, label_inds)
            neighbors.append(target_neighbors)
            # sum outer products
            dfG = _sum_outer_products(X, target_neighbors.flatten(),
                                      np.repeat(np.arange(X.shape[0]), k))

            dfGs.append(dfG)
            # initialize L
            L = self.components_

            # first iteration: we compute variables (including objective and gradient)
            #  at initialization point
            G, objective, total_active = self._loss_grad(X, L, dfG, k,
                                                         reg, target_neighbors,
                                                         label_inds)
            grads += G
            objectives += objective
        # grads /= delta
        L = self.components_

        for it in range(2, self.max_iter):
            # then at each iteration, we try to find a value of L that has better
            # objective than the previous L, following the gradient:
            while True:
                # the next point next_L to try out is found by a gradient step
                L_next = L - learn_rate * grads
                grads_next = np.zeros((feature, feature))
                objectives_next = 0
                # we compute the objective at next point
                # we copy variables that can be modified by _loss_grad, because if we
                # retry we don t want to modify them several times
                for block_ in range(delta):
                    X, y = db[block_][:, :-1], db[block_][:, -1]
                    dfG = dfGs[block_]
                    target_neighbors = neighbors[block_]
                    unique_labels, label_inds = np.unique(y, return_inverse=True)
                    (G_next, objective_next, total_active_next) = (
                        self._loss_grad(X, L_next, dfG, k, reg, target_neighbors,
                                        label_inds))
                    grads_next += G_next
                    objectives_next += objective_next
                    # break
                # grads_next /= delta

                delta_obj = objectives_next - objectives
                if delta_obj > 0:
                    # if we did not find a better objective, we retry with an L closer to
                    # the starting point, by decreasing the learning rate (making the
                    # gradient step smaller)
                    learn_rate /= 2
                else:
                    # otherwise, if we indeed found a better obj, we get out of the loop
                    break
            # when the better L is found (and the related variables), we set the
            # old variables to these new ones before next iteration and we
            # slightly increase the learning rate
            L = L_next
            grads, objectives = grads_next, objectives_next
            learn_rate *= 1.01


            if self.verbose:
                print(it, objectives, delta_obj, learn_rate)

            # check for convergence
            if it > self.min_iter and abs(delta_obj) < self.convergence_tol:
                # print(it)
                # print(objectives_next)
                if self.verbose:
                    print("LMNN converged with objective", objectives)
                break
        else:
            if self.verbose:
                print("LMNN didn't converge in %d steps." % self.max_iter)
            print('not converge')
        # store the last L
        self.components_ = L
        self.db = db
        return self

    def predict(self, X):
        db = self.db
        k = self.k
        knn = KNeighborsClassifier(n_neighbors=k)
        min_length = len(db)
        prediction_temp = []
        for block in db:
            X1 = block[:, :-1]
            y1 = block[:, -1].astype(int)
            knn.fit(self.transform(X1), y1)
            prediction_temp.append(knn.predict(self.transform(X)))
        prediction = np.vstack(prediction_temp)
        prediction = prediction.T.astype(int)
        maj = np.apply_along_axis(lambda x: np.bincount(x, minlength=min_length), axis=1, arr=prediction)
        y_pred = np.argmax(maj, axis=1)
        return y_pred

    def neigh_predict(self, X):
        db = self.db
        k = self.k
        knn = KNeighborsClassifier(n_neighbors=k)
        prediction_temp = []
        for block in db:
            X1 = block[:, :-1]
            y1 = block[:, -1].astype(int)
            knn.fit(self.transform(X1), y1)
            inds = knn.kneighbors(self.transform(X), return_distance=False)
            labels = y1[inds]
            prediction_temp.append(np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=labels))
        prediction = np.sum(prediction_temp, axis=0)
        y_pred = np.argmax(prediction, axis=1)
        return y_pred

    def predict_proba(self, X):
        db = self.db
        k = self.k
        knn = KNeighborsClassifier(n_neighbors=k)
        min_length = len(db)
        prediction_temp = []
        for block in db:
            X1 = block[:, :-1]
            y1 = block[:, -1]
            knn.fit(self.transform(X1), y1)
            prediction_temp.append(knn.predict_proba(self.transform(X)))
        proba = sum(prediction_temp) / min_length
        return proba

    def transform(self, X):
        return X.dot(self.components_.T)

    def overall_predict(self, X):
        db = self.db
        k = self.k
        knn = KNeighborsClassifier(n_neighbors=k * 10)
        knn.fit(self.transform(X))

    def sum(self, X, y, L):
        k = self.k
        reg = self.regularization
        learn_rate = self.learn_rate

        X, y = self._prepare_inputs(X, y, dtype=float,
                                    ensure_min_samples=2)
        num_pts, d = X.shape
        output_dim = _check_n_components(d, self.n_components)
        unique_labels, label_inds = np.unique(y, return_inverse=True)
        if len(label_inds) != num_pts:
            raise ValueError('Must have one label per point.')
        self.labels_ = np.arange(len(unique_labels))
        required_k = np.bincount(label_inds).min()
        if self.k > required_k:
            raise ValueError('not enough class labels for specified k'
                             ' (smallest class has %d)' % required_k)

        target_neighbors = self._select_targets(X, label_inds)

        # sum outer products
        dfG = _sum_outer_products(X, target_neighbors.flatten(),
                                  np.repeat(np.arange(X.shape[0]), k))

        # initialize L

        # first iteration: we compute variables (including objective and gradient)
        #  at initialization point
        G, objective, total_active = self._loss_grad(X, L, dfG, k,
                                                     reg, target_neighbors,
                                                     label_inds)
        return objective

    def _loss_grad(self, X, L, dfG, k, reg, target_neighbors, label_inds):
        # Compute pairwise distances under current metric
        Lx = L.dot(X.T).T

        # we need to find the furthest neighbor:
        Ni = 1 + _inplace_paired_L2(Lx[target_neighbors], Lx[:, None, :])
        temp = Ni[:, -1].copy()
        furthest_neighbors = np.take_along_axis(target_neighbors,
                                                Ni.argmax(axis=1)[:, None], 1)
        impostors = self._find_impostors(furthest_neighbors.ravel(), X,
                                         label_inds, L)


        g0 = _inplace_paired_L2(*Lx[impostors])

        # we reorder the target neighbors
        g1, g2 = Ni[impostors]
        # compute the gradient
        total_active = 0
        df = np.zeros((X.shape[1], X.shape[1]))
        for nn_idx in reversed(range(k)):  # note: reverse not useful here
            act1 = g0 < g1[:, nn_idx]
            act2 = g0 < g2[:, nn_idx]
            total_active += act1.sum() + act2.sum()

            targets = target_neighbors[:, nn_idx]
            PLUS, pweight = _count_edges(act1, act2, impostors, targets)
            df += _sum_outer_products(X, PLUS[:, 0], PLUS[:, 1], pweight)

            in_imp, out_imp = impostors
            df -= _sum_outer_products(X, in_imp[act1], out_imp[act1])
            df -= _sum_outer_products(X, in_imp[act2], out_imp[act2])

        # do the gradient update
        assert not np.isnan(df).any()
        G = dfG * reg + df * (1 - reg)
        G = L.dot(G)
        # compute the objective function
        objective = total_active * (1 - reg)
        objective += G.flatten().dot(L.flatten())
        return 2 * G, objective, total_active

    def _select_targets(self, X, label_inds):
        target_neighbors = np.empty((X.shape[0], self.k), dtype=int)
        for label in self.labels_:
            inds, = np.nonzero(label_inds == label)
            dd = euclidean_distances(X[inds], squared=True)
            np.fill_diagonal(dd, np.inf)
            nn = np.argsort(dd)[..., :self.k]
            target_neighbors[inds] = inds[nn]
        return target_neighbors

    def _find_impostors(self, furthest_neighbors, X, label_inds, L):
        Lx = X.dot(L.T)
        margin_radii = 1 + _inplace_paired_L2(Lx[furthest_neighbors], Lx)
        impostors = []
        for label in self.labels_[:-1]:
            in_inds, = np.nonzero(label_inds == label)
            out_inds, = np.nonzero(label_inds > label)
            dist = euclidean_distances(Lx[out_inds], Lx[in_inds], squared=True)
            i1, j1 = np.nonzero(dist < margin_radii[out_inds][:, None])
            i2, j2 = np.nonzero(dist < margin_radii[in_inds])
            i = np.hstack((i1, i2))
            j = np.hstack((j1, j2))
            if i.size > 0:
                # get unique (i,j) pairs using index trickery
                shape = (i.max() + 1, j.max() + 1)
                tmp = np.ravel_multi_index((i, j), shape)
                i, j = np.unravel_index(np.unique(tmp), shape)
            impostors.append(np.vstack((in_inds[j], out_inds[i])))
        if len(impostors) == 0:
            # No impostors detected
            return impostors
        return np.hstack(impostors)


def _inplace_paired_L2(A, B):
    '''Equivalent to ((A-B)**2).sum(axis=-1), but modifies A in place.'''
    A -= B
    return np.einsum('...ij,...ij->...i', A, A)


def _count_edges(act1, act2, impostors, targets):
    imp = impostors[0, act1]
    c = Counter(zip(imp, targets[imp]))
    imp = impostors[1, act2]
    c.update(zip(imp, targets[imp]))
    if c:
        active_pairs = np.array(list(c.keys()))
    else:
        active_pairs = np.empty((0, 2), dtype=int)
    return active_pairs, np.array(list(c.values()))


def _sum_outer_products(data, a_inds, b_inds, weights=None):
    Xab = data[a_inds] - data[b_inds]
    if weights is not None:
        return np.dot(Xab.T, Xab * weights[:, None])
    return np.dot(Xab.T, Xab)


def error_compute(db, tr, i):
    knn = KNeighborsClassifier(n_neighbors=3)
    error = 0
    count = 0
    for j in range(len(db)):
        if i == j:
            continue
        else:
            X, y = db[j][:, :-1], db[j][:, -1]
            X_tr = X.dot(tr.T)
            knn.fit(X_tr, y)
            y_pred = knn.predict(X_tr)
            error += np.sum(y_pred != y)
            count += len(y)
    error_rate = error / count
    error_rate = 0.5 * np.log((1 - error_rate) / error_rate)
    return error_rate





