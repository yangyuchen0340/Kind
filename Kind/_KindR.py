#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:19:55 2019

@author: yangyc
"""

# %% library
import numpy as np
from scipy import linalg as la
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import normalize
from six import string_types
from sklearn import cluster
from .utils import proj_ud, proj_h
from ._KindAP import kindap

import warnings


# %% KindR
def kindr(Ud, n_clusters, init, tol_in, tol_out, max_iter_in, max_iter_out, disp,
          do_inner, post_SR, isnrm_row_U, isnrm_col_H, isbinary_H):
    if max_iter_out <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter_out)
    if tol_out <= 0:
        raise ValueError('The tolerance should be a positive number,'
                         ' got %d instead' % tol_out)
    try:
        from pymanopt import Problem
        from pymanopt.manifolds import Stiefel, Rotations
        from pymanopt.solvers import SteepestDescent
    except ImportError:
        raise ValueError("KindR needs pymanopt, which is unavailable, try KindAP instead.")
    #                warnings.warn("KindR solver is unavailable. Transfer to"
    #                                 "KindAP instead.")
    try:
        import autograd.numpy as anp
    except ImportError:
        warnings.warn("Pymanopt needs autograd, which is unavailable,try KindAP instead.")
        idx, center, gerr, numiter = kindap(Ud, n_clusters, init, tol_in, tol_out, max_iter_in, max_iter_out,
                                            disp, True, post_SR, isnrm_row_U, isnrm_col_H, isbinary_H)
        return idx, center, gerr, numiter

    n, d = Ud.shape
    k = n_clusters
    if d != k:
        warnings.warn('Provided more features, expected %d, got %d' % (k, d))
    if isnrm_row_U:
        Ud = normalize(Ud, axis=1)
    # initialization
    if isinstance(init, string_types) and init == 'eye':
        # Z_0 = -np.identity(k)
        U = Ud[:, :k]
    elif hasattr(init, '__array__'):
        U = np.array(init)
        if U.shape[0] != n:
            raise ValueError('The row size of init should be the same as the total'
                             'observations, got %d instead.' % U.shape[0])
        if U.shape[1] != k:
            raise ValueError('The column size of init should be the same as the total'
                             'clusters, got %d instead.' % U.shape[1])
    elif isinstance(init, string_types) and init == 'random':
        H = sparse.csc_matrix((np.ones((n,)), (np.arange(n), np.random.randint(0, k, (n,)))), shape=(n, k))
        S, D, V = la.svd(sparse.csc_matrix.dot(Ud.T, H), full_matrices=False)
        Z_0 = np.matmul(S, V)
        U = np.matmul(Ud, Z_0)
    else:
        raise ValueError("The init parameter for KindAP should be 'eye','random',"
                         "or an array. Got a %s with type %s instead." % (init, type(init)))
    if isinstance(do_inner, bool) or isinstance(do_inner, int):
        do_inner = bool(do_inner)
    elif isinstance(do_inner, string_types) and (do_inner in ["relu", "softmax"]):
        pass
    else:
        raise ValueError("Invalid put do_inner")

    H, N = sparse.csc_matrix((n, k)), sparse.csc_matrix((n, k))
    dUH = float('inf')
    numiter, gerr = [], []
    idx = np.ones(n)
    crit2 = np.zeros(4)

    def cost_n(rotation):
        return 0.5 * anp.sum(anp.minimum(anp.matmul(Ud, rotation), 0) ** 2)

    def cost_softmax(rotation):
        umat = anp.matmul(Ud, rotation)
        exp_umat = anp.exp(umat)
        mat = exp_umat / exp_umat.sum(axis=1).reshape(Ud.shape[0], 1)
        return 0.5 * anp.sum((umat - mat) ** 2)

    for n_iter_out in range(max_iter_out):
        idxp, Up, Np, Hp = idx, U, N, H
        ci = -1
        itr = 0

        if disp:
            print("Outer Iteration %3d: " % (n_iter_out + 1))
        # inner iterations
        if do_inner:
            if d == k:
                manifold = Rotations(k)
            else:
                manifold = Stiefel(d, k)
            if do_inner == "softmax":
                cost = cost_softmax
            else:
                cost = cost_n
            problem = Problem(manifold=manifold, cost=cost, verbosity=0)
            solver = SteepestDescent(maxiter=max_iter_in, mingradnorm=tol_in, logverbosity=2)
            Z, optlog = solver.solve(problem)
            U = np.matmul(Ud, Z)
            N = np.maximum(U, 0)
            numiter.append(len(optlog['iterations']['iteration']))
            if disp:
                print(optlog['iterations']['f(x)'])
        else:
            N = U
            numiter.append(0)

        # project onto H            
        H, idx = proj_h(N, isnrm_col_H, isbinary_H)
        idxchg = la.norm(idx - idxp, 1)

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
                print('\t stop reason:', optlog['stoppingreason'])
            break
    center = sparse.csc_matrix.dot(normalize((H != 0), axis=0, norm='l1').T, Ud)
    return idx, center, gerr, numiter


class KindR(BaseEstimator, ClusterMixin, TransformerMixin):
    """ K-indicators model with Riemann optimization algorithm
    Author: Yuchen Yang, Yin Zhang
    """

    def __init__(self, n_clusters, init='eye', tol_in=1e-3,
                 tol_out=1e-6, max_iter_in=200, max_iter_out=50, disp=False,
                 do_inner=True, post_SR=True, isnrm_row_U=False, isnrm_col_H=True,
                 isbinary_H=False):
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

        k = int(self.n_clusters)
        if k > X.shape[1]:
            raise ValueError("KindR can only solve cases where dim > n_clusters")

        self.labels_, self.cluster_centers_, err, self.n_iter = \
            kindr(X, k, self.init, self.tol_in, self.tol_out, self.max_iter_in, self.max_iter_out,
                  self.disp, self.do_inner, self.post_SR, self.isnrm_row_U, self.isnrm_col_H,
                  self.isbinary_H)
        if len(err) > 0:
            self.inertia_ = err[-1]
        else:
            raise ValueError("Insufficient error array.")
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def fit_L(self, X):
        self.fit(X)
        k = int(self.n_clusters)
        if k > X.shape[0]:
            raise ValueError("Clustering algorithms have more clusters than observations")
        if k > X.shape[1]:
            raise ValueError("KindAP can only deal with d>=k as inputs, may need another type of preprocessing")
        km = cluster.KMeans(n_clusters=k, init=self.cluster_centers_)
        km.fit(X)
        self.labels_ = km.labels_
        self.cluster_centers_ = km.cluster_centers_
        self.inertia_ = km.inertia_
        return self

    def fit_predict_L(self, X):
        self.fit_L(X)
        return self.labels_
