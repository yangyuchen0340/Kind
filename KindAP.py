#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:22:33 2018

@author: yangyc
"""

# K-indicators with two-layered alternating projection algorithm
# Yuchen Yang 
# prototype version of KindAP

#%% library
import numpy as np
from scipy import linalg as la
from scipy.stats import ortho_group
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import normalize
from six import string_types
from sklearn import cluster
import warnings

#%% Inner iteration
def nonneg_proj(Uk,Z,tol,max_iter,if_print):
    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)
    if tol <= 0:
        raise ValueError('The tolerance should be a positive number,'
                         ' got %d instead' % tol)
    n,k = Uk.shape
    N = np.zeros((n,k))
    U = np.matmul(Uk,Z)
    err_new = -1
    err = []
#    max_iter = 1000 # can be put in init section in the future
#    tol = 1e-4    
    itr = 1
    while itr < max_iter:
#        U_old = U
        N_old = N
        N = np.maximum(U,0)
        # Here, the sign flip is to be added
        if (itr%5==0):
            err_old = err_new # This does not need to run every time
            err_new = la.norm(U-N,'fro')
            err.append(err_new)
            crt_1 = la.norm(N-N_old,'fro')<tol
            crt_2 = abs(err_new-err_old)<tol
            if if_print == True :
                print ("Inner Step ", itr, " with residual ", err_new)
            if (crt_1 or crt_2):
                break
        S,D,V=la.svd(np.dot(Uk.transpose(),N),full_matrices=False)
        Z = np.matmul(S,V)
        U = np.matmul(Uk,Z)
        # Accelerate SVD part: randomized or truncated and fixed
#        if itr > 2 :
#            t=(np.trace(np.dot(U_old.T,U_old-U))+np.trace(np.dot(N_old.T,N_old-N))) / (la.norm(N-N_old,'fro')**2+la.norm(U-U_old,'fro')**2)
#            N = N_old*t+N*(1-t)
#            U = U_old*t+U*(1-t)
        itr = itr+1
    return N,Z,err,itr

#%% KindAP draft
def kindAP(Uk,init,tol_in,tol_out,max_iter_in, max_iter_out,if_print):# row version only
    if max_iter_out <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter_out)
    if tol_out <= 0:
        raise ValueError('The tolerance should be a positive number,'
                         ' got %d instead' % tol_out)
        
    n,k = Uk.shape
    tol = tol_out
    gerr = []
    n_iter_in = 0
    if isinstance(init, string_types) and init == 'eye':
        Z_0 = -np.identity(k)
    elif isinstance(init, string_types) and init == 'random':
        Z_0 = ortho_group.rvs(dim=k)
    elif hasattr(init, '__array__'):
        J = np.array(init) - min(init) # may need label encoder
        if J.shape[0] != n :
            raise ValueError('The size of init should be the same as the total'
                             'observations, got %d instead.' % J.shape[0])
        if np.unique(J).shape[0] < k:
            warnings.warn('The init should contain the required number of'
                             'clusters %d. Got %d instead.'% (k,np.unique(J).shape[0]))
        elif np.unique(J).shape[0] > k:
            raise ValueError('The init should contain the required number of'
                             'clusters %d. Got %d instead.'% (k,np.unique(J).shape[0]))
        I = np.arange(n)
        V = np.ones((n,))
        H = sparse.csc_matrix((V,(I,J)),shape=(n,k))
        S,D,V=la.svd(sparse.csc_matrix.dot(Uk.transpose(),H),full_matrices=False)
        Z_0 = np.matmul(S,V)
    else:
        raise ValueError("the init parameter for KindAP should be 'eye','random',"
                         "or an array. Got a %s with type %s instead."%(init, type(init)))
    H = sparse.csc_matrix((n,k))
    for n_iter_out in range(max_iter_out):
        Z = Z_0
        if if_print == True :
            print ("Outer Iteration : ",n_iter_out+1, "\n")
        N,Z,err,itr = nonneg_proj(Uk,Z,tol_in,max_iter_in, if_print)
        n_iter_in = n_iter_in + itr
        if n_iter_out > 0:
            H_last = H
        I = np.arange(n)
        J = np.array(np.argmax(N,axis=1)).reshape((n,))
        V = np.array(np.max(N,axis=1)).reshape((n,))
        H = sparse.csc_matrix((V,(I,J)),shape=(n,k))
        # normalization to be added
        res = la.norm(np.matmul(Uk,Z)-H,'fro')
        gerr.append(res)
        # stopping criteria can be refined
        if (n_iter_out>0) and (res>gerr[-2]-tol):
            break
        S,D,V=la.svd(sparse.csc_matrix.dot(Uk.transpose(),H),full_matrices=False)
        Z_0 = np.matmul(S,V)
    if res > gerr[-2]:
        H = H_last
    idx = np.argmax(H,axis=1)[:,0]
    idx = np.asarray(idx).reshape(-1)+1
    return idx, H, gerr[-2], n_iter_in, n_iter_out+1


class KindAP(BaseEstimator, ClusterMixin, TransformerMixin):
    """ K-indicators model with alternating projection algorithm
    Author: Yuchen Yang, Feiyu Chen, Yin Zhang
    """
    
    def __init__(self, n_clusters=10, init = 'eye', algorithm = 'simple', tol_in=1e-4, 
                 tol_out=1e-6, max_iter_in=1000, max_iter_out=1000, if_print=False):
        self.n_clusters = n_clusters
        self.init = init
        self.algorithm = algorithm
        self.max_iter_in = max_iter_in
        self.tol_in = tol_in
        self.tol_out = tol_out
        self.max_iter_out = max_iter_out 
        self.if_print = if_print

    
    def fit(self,X):
        if self.algorithm not in ['simple','L']:
            raise ValueError("Invalid algorithm, KindAP should be 'simple','L'," 
                             "'outlier',or 'full'. Got a %s instead" % self.algorithm)
        k = int(self.n_clusters)
        if k > X.shape[1]:
            raise ValueError("KindAP can only solve cases where dim>n_clusters")
        S,D,V=la.svd(X,full_matrices=False)
        Uk = S[:,:k]
        self.labels_, H, self.inertia_, self.n_iter_in, self.n_iter_out = \
        kindAP(Uk, self.init, self.tol_in, self.tol_out,self.max_iter_in,self.max_iter_out, 
               self.if_print)
        H = normalize((H!=0),axis=0,norm='l1').transpose()
        self.cluster_centers_ = np.matmul(H.todense(),X)
        if self.algorithm == 'simple':
            return self
        else:
            self.algorithm == 'L'
            km = cluster.KMeans(n_clusters=k,init=self.cluster_centers_)
            km.fit(X)
            self.labels_ = km.labels_
            self.cluster_centers_ = km.cluster_centers_
            self.inertia_ = km.inertia_
            return self
        
            
    