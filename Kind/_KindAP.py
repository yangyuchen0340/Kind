#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:22:33 2018

@author: yangyc
"""

# K-indicators with two-layered alternating projection algorithm
# Yuchen Yang 
# prototype version of KindAP

# %% library
import numpy as np
from scipy import linalg as la
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import normalize
from six import string_types
from sklearn import cluster
import warnings
from .utils import proj_h, proj_ud


# %% KindAP draft
def kindap(Ud, n_clusters, init, tol_in, tol_out, max_iter_in, max_iter_out, disp,
           do_inner, post_SR, isnrm_row_U, isnrm_col_H, isbinary_H):  # row version only
    """
    run KindAP algorithm, Yuchen Yang, Feiyu Chen, Yin Zhang
    :param Ud: Input n*d matrix data
    :param n_clusters: the number of clusters estimated
    :param init: the initial value, which can be a given array, or 'random' or 'eye'
    :param tol_in: the tolerance of inner iterations
    :param tol_out: the tolerance of outer iterations
    :param max_iter_in: the maximum number of inner iterations
    :param max_iter_out: the maximum number of outer iterations
    :param disp: display Bool: the iterative results or not
    :param do_inner: Bool: do the inner iterations or not
    :param post_SR: Bool: do the post spectral rotation or not
    :param isnrm_row_U: Bool: normalize Ud row-wise or not
    :param isnrm_col_H: Bool: normalize H column-wise or not
    :param isbinary_H: Bood: use binary entries of H or not
    :return:
    """
    if max_iter_out <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter_out)
    if tol_out <= 0:
        raise ValueError('The tolerance should be a positive number,'
                         ' got %d instead' % tol_out)

    n, d = Ud.shape
    k = n_clusters
    if d != k:
        warnings.warn('Provided more features: expected %d, got %d' % (k, d))
    if isnrm_row_U:
        Ud = normalize(Ud, axis=1)
    # initialization
    if isinstance(init, string_types) and init == 'eye':
        # Z_0 = -np.identity(k)
        U = Ud[:, :k]
    elif hasattr(init, '__array__'):
        if init.shape[0] != d:
            raise ValueError('The row size of init should be the same as the total'
                             'features, got %d instead.' % init.shape[0])
        if init.shape[1] != k:
            raise ValueError('The column size of init should be the same as the total'
                             'clusters, got %d instead.' % init.shape[1])
        U = np.matmul(Ud, np.array(init))
    elif isinstance(init, string_types) and init == 'random':
        H = sparse.csc_matrix((np.ones((n,)), (np.arange(n), np.random.randint(0, k, (n,)))), shape=(n, k))
        smat, sigma, vmat = la.svd(sparse.csc_matrix.dot(Ud.T, H), full_matrices=False)
        z_init = np.matmul(smat, vmat)
        U = np.matmul(Ud, z_init)
    #    elif isinstance(init, string_types) and init == 'adaptive':
    else:
        raise ValueError("the init parameter for KindAP should be 'eye','random',"
                         "or an array. Got a %s with type %s instead." % (init, type(init)))
    H, N = sparse.csc_matrix((n, k)), sparse.csc_matrix((n, k))
    dUH = float('inf')
    numiter, gerr = [], []
    idx = np.ones(n)
    crit1, crit2 = np.zeros(3), np.zeros(4)

    for n_iter_out in range(max_iter_out):
        idxp, Up, Np, Hp = idx, U, N, H
        dUN = float('inf')
        ci = -1
        itr = 0

        if disp:
            print("\nOuter Iteration %3d: " % (n_iter_out + 1))
        # inner iterations
        if do_inner:
            for itr in range(max_iter_in):
                N = np.maximum(U, 0)
                U = proj_ud(N, Ud)
                dUNp = dUN
                dUN = la.norm(U - N, 'fro')
                if disp:
                    print('iter: %4d, dUN = %11.8e' % (itr + 1, dUN))
                # stopping criteria of inner iterations
                crit1[0] = dUN < 1e-12
                crit1[1] = abs(dUN - dUNp) < dUNp * tol_in
                crit1[2] = dUN > dUNp
                if any(crit1):
                    numiter.append(itr)
                    ci = np.nonzero(crit1)[0][0]
                    break
        else:
            N = U
            numiter.append(0)

        # project onto H            
        H, idx = proj_h(N, isnrm_col_H, isbinary_H)
        idxchg = sum(idx != idxp)

        # project back to Ud
        U = proj_ud(H, Ud)
        dUHp = dUH
        dUH = la.norm(U - H, 'fro')
        gerr.append(dUH)
        if disp:
            print('Outer %3d: %3d(%d)  dUH: %11.8e idxchg: %6d' % (n_iter_out + 1, itr, ci, dUH, idxchg))
        # stopping criteria of outer iterations
        crit2[0] = dUH < 1e-12
        crit2[1] = abs(dUH - dUHp) < dUHp * tol_out
        crit2[2] = dUH > dUHp
        crit2[3] = idxchg == 0
        if any(crit2):
            if post_SR and do_inner:
                do_inner, isbinary_H = False, True
                continue
            if crit2[2] and not crit2[1]:
                idx, H, U, N, dUH = idxp, Hp, Up, Np, dUHp
            if disp:
                print('\t stop criteria:', np.nonzero(crit1)[0])
            break
    center = sparse.csc_matrix.dot(normalize((H != 0), axis=0, norm='l1').T, Ud)
    return idx, center, gerr, numiter


class KindAP(BaseEstimator, ClusterMixin, TransformerMixin):
    """ K-indicators model with alternating projection algorithm
    Author: Yuchen Yang, Feiyu Chen, Yin Zhang
    """

    def __init__(self, n_clusters, init='eye', tol_in=1e-3,
                 tol_out=1e-5, max_iter_in=200, max_iter_out=50, disp=False,
                 do_inner=True, post_SR=True, isnrm_row_U=True, isnrm_col_H=False,
                 isbinary_H=True):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter_in = max_iter_in
        self.max_iter_out = max_iter_out
        self.tol_in = tol_in
        self.tol_out = tol_out
        self.disp = disp
        self.isnrm_row_U = isnrm_row_U
        self.isnrm_col_H = isnrm_col_H
        self.isbinary_H = isbinary_H
        self.do_inner = do_inner
        self.post_SR = post_SR
        self.labels_ = []
        self.n_iter = 0
        self.inertia_ = float('inf')
        self.cluster_centers_ = []

    def fit(self, X):
        """
        fit the data
        :param X: the input preprocessed data
        :return: self: class KindAP
        """
        k = int(self.n_clusters)
        if k > X.shape[0]:
            raise ValueError("Clustering algorithms have more clusters than observations")
        if k > X.shape[1]:
            raise ValueError("KindAP can only deal with d>=k as inputs, may need another type of preprocessing")

        self.labels_, self.cluster_centers_, err, self.n_iter = \
            kindap(X, k, self.init, self.tol_in, self.tol_out, self.max_iter_in, self.max_iter_out,
                   self.disp, self.do_inner, self.post_SR, self.isnrm_row_U, self.isnrm_col_H,
                   self.isbinary_H)
        if len(err) > 0:
            self.inertia_ = err[-1]
        else:
            raise ValueError('Insufficient error array.')
        return self

    def fit_predict(self, X, y=None):
        """
        fit the data and predict the labels
        :param X: the input preprocessed data
        :param y: Ignored for unsupervised learning
        :return: predicted labels
        """
        self.fit(X)
        return self.labels_

    def fit_L(self, X):
        """
        fit the data with KindAP+L
        :param X: the input preprocessed data
        :return: self
        """
        self.fit(X)
        k = int(self.n_clusters)
        if k > X.shape[1]:
            raise ValueError("n_clusters is greater than the number of observations")
        km = cluster.KMeans(n_clusters=k, init=self.cluster_centers_)
        km.fit(X)
        self.labels_ = km.labels_
        self.cluster_centers_ = km.cluster_centers_
        self.inertia_ = km.inertia_
        return self

    def fit_predict_L(self, X):
        self.fit_L(X)
        return self.labels_
