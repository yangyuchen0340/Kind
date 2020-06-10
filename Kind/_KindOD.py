import numpy as np
from scipy import linalg as la
from scipy import sparse
from ._KindAP import KindAP
from six import string_types
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import warnings


def prox_l2(tmat, mu):
    """
    Solve soft thresholding with l_{2,1} norm
    original: argmin_{X in R^{n*k}} 1/2||X-T||_2^2+mu||X||_{2，1}
    row-wise: argmin_{x in R^k} 1/2||x-t||_2^2+mu||x||2
    Reference: A fast algorithm for edge-preserving variational multichannel image restoration  (Lemma 3.3)
    Copyright: Yuchen Yang

    :param tmat: input matrix
    :param mu: parameter adjusting the strength of outliers
    :return: xmat, the proximal of T
    """

    n, k = tmat.shape
    xmat = np.zeros(shape=(n, k))
    for i in range(n):
        t = tmat[i, :]
        nrm = la.norm(t, 2)
        if nrm <= mu:
            x = np.zeros(k)
        else:
            x = (1 - mu / nrm) * t
        xmat[i, :] = x
    return xmat


def kindod(umat, n_clusters, mu, disp, maxit, tol_rel, tol_abs, isnrm_col_H, isnrm_row_U, isbinary_H):
    """
    K-indicators with outlier detections, author: Yuchen Yang
    :param isnrm_col_H:
    :param isnrm_row_U:
    :param isbinary_H:
    :param umat: input data, currently supports n*n_clusters
    :param n_clusters: the number of clusters estimated
    :param mu: mu can be a given real non-negative number, a 'XX%' percentage, a 'Furthest X' number of points,
                or simply adaptive. Default = 'adaptive'
    :param disp: display the results or not. Default = False
    :param maxit: the maximum number of iterations. Default maxit = 100
    :param tol_rel: relative tolerance. Default tol_rel = 1e-6
    :param tol_abs: absolute tolerance. Default tol_abs = 1e-3
    :return:
    idx: clustering results
    idc: the id of outliers
    center: cluster centers
    lagrangian: objectives of lagrangian function
    primal: primal residuals
    dual: dual residuals
    W: corrected data
    """
    n, d = umat.shape
    if maxit <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % maxit)
    if tol_rel <= 0 or tol_abs <= 0:
        raise ValueError('Relative and absolute tolerance should both be positive,'
                         ' got tol_rel = %f, and tol_abs = %f instead' % (tol_rel, tol_abs))
    num_parameter = perc_parameter = -1
    if isinstance(mu, int) or isinstance(mu, float):
        mu = float(mu)
        if mu < 0:
            raise ValueError('Requires positive mu, got %f instead' % mu)
    elif isinstance(mu, string_types) and len(mu) and mu[-1] == '%' and mu[:-1].isnumeric() and 0 <= float(
            mu[:-1]) <= 100:
        perc_parameter = float(mu[:-1]) / 100
    elif isinstance(mu, string_types) and len(mu) >= 9 and mu[:8].lower() == 'furthest' and mu[9:].isnumeric() \
            and 0 <= int(mu[9:]) <= n:
        num_parameter = int(mu[9:])
    elif isinstance(mu, string_types) and mu.lower() == 'adaptive':
        perc_parameter = -2
    elif isinstance(mu, string_types):
        raise ValueError('Invalid mu: (must be ??% or Furthest ?? or adaptive), got %s instead' % mu)
    else:
        raise ValueError('Unknown mu input')
    idx, lagrangian, primal, dual = [], [], [], []
    if isnrm_row_U:
        umat = normalize(umat, axis=1)
    rho = 1
    W, Wp = umat, np.zeros(shape=(n, d))
    V = L = np.zeros(shape=(n, d))
    if d > n_clusters:
        Z = np.concatenate(np.identity(n_clusters), np.zeros(shape=(d - n_clusters, n_clusters)), axis=0)
    elif d == n_clusters:
        Z = np.identity(n_clusters)
    else:
        raise ValueError('Insufficient features, get %d but need %d at least' % (d, n_clusters))
    centers = np.zeros(shape=(n_clusters, d))  # will be k*d in the future
    for i in range(maxit):
        # J = np.argmax(np.abs(V), axis=1)
        # H = sparse.csc_matrix((np.ones((n,)), (np.arange(n), J)), shape=(n, n_clusters))
        # if isnrm_col_H:
        #     H = normalize(H, axis=0)
        # s, sigma, v = la.svd(sparse.csc_matrix.dot(W.T, H), full_matrices=False)
        # Z = np.matmul(s, v)
        ki = KindAP(n_clusters=n_clusters, init=Z, max_iter_out=2, isnrm_row_U=isnrm_row_U, isnrm_col_H=isnrm_col_H,
                    isbinary_H=isbinary_H)
        ki.fit(W)
        idx, centers = ki.labels_, ki.cluster_centers_
        H = sparse.csc_matrix((np.ones((n,)), (np.arange(n), idx.flatten())), shape=(n, n_clusters))
        if isnrm_col_H:
            H = normalize(H, axis=0)
        s, sigma, v = la.svd(sparse.csc_matrix.dot(W.T, H), full_matrices=False)
        Z = np.matmul(s, v)
        B = W - umat
        if num_parameter != -1:
            temp = la.norm(B + L / rho, axis=1)
            mu = np.partition(temp, -num_parameter)[-num_parameter]  # Select k-th largest values
            mu = mu * rho / 2.0001
        elif perc_parameter == -2:
            temp = la.norm(B + L / rho, axis=1)
            temp = np.partition(temp, -int(0.1 * n))[-int(0.1 * n):]  # the maximum percentage of outliers
            temp = np.sort(temp)[::-1]  # sort the row norm in descending order
            if len(temp) >= 3:
                curv = temp[:-2] + temp[2:] - 2 * temp[1:-1]  # second-order finite-difference
                if len(np.where(curv > 0)[0]) > 0:
                    thres_id = np.argmax(curv)  # the location of maximal decrease
                else:
                    thres_id = len(temp) - 1
                mu = temp[thres_id] * rho / 2.0001
            elif len(temp) >= 1:
                thres_id = len(temp) - 1
                mu = temp[thres_id] * rho / 2.0001
            else:
                mu = 0
        elif perc_parameter != -1:
            temp = la.norm(B + L / rho, axis=1)
            mu = np.partition(temp, -int(perc_parameter * n))[-int(perc_parameter * n)]
            mu = mu * rho / 2.0001
        V = prox_l2(B + L / rho, 2 * mu / rho)
        A = V + umat
        #        s, sigma, v = la.svd(rho * A - L + sparse.csc_matrix.dot(H, Z.T), full_matrices=False)
        #        W = np.matmul(s, v)
        W = (rho * A - L + sparse.csc_matrix.dot(H, Z.T)) / (1 + rho)
        L = L + rho * (W - A)
        S = -rho * (W - Wp)
        Wp = W
        obj = 0.5 * la.norm(np.matmul(W, Z) - H, 'fro') + mu * sum(la.norm(V, axis=1)) \
            + np.trace(np.dot(L.T, W - A)) + rho / 2 * sum(sum(abs(W - A) ** 2))
        prim_res, dual_res = la.norm(W - A, 'fro'), la.norm(S, 'fro')
        primal.append(prim_res)
        dual.append(dual_res)
        lagrangian.append(obj)
        prim_feasible = prim_res <= np.sqrt(n * d) * tol_abs + max(d, la.norm(V, 'fro')) * tol_rel
        dual_feasible = dual_res <= np.sqrt(n * d) * tol_abs + la.norm(L, 'fro') * tol_rel
        if disp:
            print("Step: %2d, Primal Residual: %1.5e, Dual Residual: %1.5e, Obj: %1.5e" % (i + 1, prim_res,
                                                                                           dual_res, obj))
        if prim_feasible or dual_feasible:
            break
        if prim_res > 10 * dual_res:
            rho *= 2
        elif prim_res < 0.1 * dual_res:
            rho /= 2
        else:
            continue
    idc = np.nonzero(np.max(abs(V), axis=1))[0]
    return idx, idc, centers, lagrangian, primal, dual, W


class KindOD(BaseEstimator, ClusterMixin, TransformerMixin):
    """ K-indicators model with outlier detections add-ons
    Author: Yuchen Yang, Feiyu Chen, Yin Zhang
    """

    def __init__(self, n_clusters, mu="adaptive", maxit=100, disp=False, tol_rel=1e-6, tol_abs=1e-3,
                 isnrm_col_H=False, isnrm_row_U=True, isbinary_H=True):
        self.disp = disp
        self.n_clusters = n_clusters
        self.maxit = maxit
        self.tol_rel = tol_rel
        self.tol_abs = tol_abs
        self.isnrm_col_H = isnrm_col_H
        self.isnrm_row_U = isnrm_row_U
        self.isbinary_H = isbinary_H
        self.mu_ = mu
        self.labels_ = []
        self.outliers_ = []
        self.cluster_centers_ = []
        self.corrected_data_ = []
        self.inertia_ = float('inf')

    def fit(self, X):
        k = int(self.n_clusters)
        n, d = X.shape
        if d < k:
            raise ValueError('Invalid X, expect more features than clusters, got %d instead' % X.shape[1])
        if n < k:
            raise ValueError('Invalid X, expect more observations than clusters, got %d instead' % X.shape[0])
        if not self.isnrm_row_U and not np.allclose(np.dot(X.T, X), np.eye(d)):
            U, R = la.qr(X, mode='economic')
            warnings.warn('Current version only supports the input data is orthonormal, transferred by QR instead.')
        else:
            U, R = X, np.eye(d)
        self.labels_, self.outliers_, self.cluster_centers_, lagrangian, primal, dual, W = kindod(U, self.n_clusters,
                                                                                                  self.mu_,
                                                                                                  self.disp, self.maxit,
                                                                                                  self.tol_rel,
                                                                                                  self.tol_abs,
                                                                                                  self.isnrm_col_H,
                                                                                                  self.isnrm_row_U,
                                                                                                  self.isbinary_H)
        self.corrected_data_ = np.dot(W, R)
        if len(lagrangian) > 0:
            self.inertia_ = lagrangian[-1]
        else:
            raise ValueError('Insufficient error array.')
        if self.disp:
            print('The primal residuals are ', primal)
            print('The dual residuals are ', dual)
            print('The Lagrangian objectives are ', lagrangian)
            print('The outliers are ', self.outliers_)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def fit_predict_outliers(self, X, y=None):
        self.fit(X)
        return self.outliers_
