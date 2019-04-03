#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:09:22 2019

@author: yangyc
"""
import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import norm
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph

from ._KindAP import KindAP
from .utils import _deterministic_vector_sign_flip


def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition
    Parameters
    ----------
    laplacian : array or sparse matrix
        The graph laplacian
    value : float
        The value of the diagonal
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not
    Returns
    -------
    laplacian : array or sparse matrix
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def kind_joint(K, n_clusters, init, maxit, disp, tol, norm_laplacian):
    if maxit <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % maxit)
    if tol <= 0:
        raise ValueError('The tolerance should be a positive number,'
                         ' got %d instead' % tol)
    if K.shape[0] != K.shape[1]:
        warnings.warn('Input is not an affinity matrix. Kernelize using KNN'
                      'graph now')
        X = kneighbors_graph(K)
    else:
        X = (K + K.T) / 2

    # set initial V
    V = spectral_embedding(X, n_components=n_clusters,
                           drop_first=False, norm_laplacian=norm_laplacian)
    # set initial idx
    n = X.shape[0]
    if hasattr(init, '__array__'):
        idx = np.array(init).reshape(max(init.shape()), )
        if idx.shape[0] != n:
            raise ValueError('The init should be the same as the total'
                             'observations, got %d instead.' % idx.shape[0])
    else:
        km = KindAP(n_clusters=n_clusters)
        idx = km.fit_predict_L(V)
    # set rho
    rho = 1 / n
    # set history info
    hist = [0 for i in range(maxit)]
    for itr in range(maxit):
        Vp, idxp = V, idx

        laplacian, dd = csgraph_laplacian(X, normed=norm_laplacian,
                                          return_diag=True)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        laplacian *= -1
        v0 = np.random.uniform(-1, 1, laplacian.shape[0])
        I = np.arange(n)
        V = np.ones(n)
        H = sparse.csc_matrix((V, (I, idx)), shape=(n, n_clusters))
        lambdas, diffusion_map = eigsh(laplacian + rho * sparse.csc_matrix.dot(H, H.T),
                                       k=n_clusters, sigma=1.0, which='LM', v0=v0)
        embedding = diffusion_map.T[n_clusters::-1]
        V = _deterministic_vector_sign_flip(embedding)
        if norm_laplacian:
            V = embedding / dd
        obj = rho * np.sum(sparse.csc_matrix.dot(V, H) ** 2) + np.trace(
            np.dot(sparse.csc_matrix.dot(V, laplacian), V.T))
        hist[itr] = 0.5 * obj
        V = V.T
        ki = KindAP(n_clusters=n_clusters)
        idx = ki.fit_predict_L(V)

        # stopping criteria
        idxchg = norm(idx - idxp, 1)
        Vrel = norm(V - Vp, 'fro') / norm(Vp, 'fro')
        if disp:
            print('iter: %3d, Obj: %6.2e,  Vrel: %6.2e, idxchg: %6d' % (itr, obj, Vrel, idxchg))
        if not idxchg or Vrel < tol:
            break

    return idx, V, hist[:min(maxit, itr + 1)]


class KindJoint(BaseEstimator, ClusterMixin, TransformerMixin):
    """ K-indicators model with Joint optimization
    Author: Yuchen Yang, Yin Zhang
    """

    def __init__(self, n_clusters, init=None, tol=1e-5, maxit=200, disp=False,
                 norm_laplacian=True):
        self.n_clusters = n_clusters
        self.init = init
        self.maxit = maxit
        self.tol = tol
        self.disp = disp
        self.norm_laplacian = norm_laplacian

    def fit(self, X):

        k = int(self.n_clusters)
        if k > X.shape[0]:
            raise ValueError("n_clusters is greater than the number of observations")

        self.labels_, self.embedding_, hist = \
            kind_joint(X, k, self.init, self.maxit, self.disp, self.tol,
                       self.norm_laplacian)
        if len(hist) > 0:
            self.inertia_ = hist[-1]
            self.iter = len(hist)
        else:
            raise ValueError("Insufficient error array.")
        return self

    def fit_predict(self, X, y=None):
        """

        :param X:
        :type y: Ignored
        """
        self.fit(X)
        return self.labels_
