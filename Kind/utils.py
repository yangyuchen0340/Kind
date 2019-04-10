#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:42:17 2019

@author: yangyc
"""
import numpy as np
from scipy import linalg as la
from scipy import sparse
from sklearn.preprocessing import normalize
from munkres import Munkres
from copy import deepcopy


def confusion_matrix(clusters, classes_gt):
	new_gt = deepcopy(classes_gt)
	l = list(set(classes_gt))
	for i in range(len(classes_gt)):
		for j in range(len(l)):
			if classes_gt[i] == l[j]:
				new_gt[i] = j

	conf_mat = np.zeros([len(set(clusters)), len(set(new_gt))])
	for i in range(len(clusters)):
		conf_mat[clusters[i], new_gt[i]] += 1

	return conf_mat


def purity(clusters, classes_gt):
	conf_mat = confusion_matrix(clusters, classes_gt)
	sum_clu = np.max(conf_mat, axis=1)
	sum_tot = np.sum(sum_clu)
	pur = sum_tot / len(clusters)

	return pur


def proj_ud(N, Ud):
	if sparse.issparse(N):
		S, D, V = la.svd(sparse.csc_matrix.dot(Ud.T, N), full_matrices=False)
	else:
		S, D, V = la.svd(np.dot(Ud.T, N), full_matrices=False)
	Z = np.matmul(S, V)
	U = np.matmul(Ud, Z)
	return U


def proj_h(N, isnrm_col_H, isbinary_H):
	n, k = N.shape
	I = np.arange(n)
	J = np.array(np.argmax(N, axis=1)).reshape((n,))
	if isbinary_H:
		V = np.array(np.max(N, axis=1)).reshape((n,))
	else:
		V = np.ones(n)
	H = sparse.csc_matrix((V, (I, J)), shape=(n, k))
	if isnrm_col_H:
		H = normalize(H, axis=0)
	return H, J


def best_map(L1, L2):
	# L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1, nClass2)
	G = np.zeros((nClass, nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i, j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:, 1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2


def _deterministic_vector_sign_flip(u):
	"""
	Modify the sign of vectors for reproducibility
	Flips the sign of elements of all the vectors (rows of u) such that
	the absolute maximum element of each vector is positive.
	Parameters
	Copyright: Sklearn
	----------
	u : ndarray
		Array with vectors as its rows.
	Returns
	-------
	u_flipped : ndarray with same shape as u
		Array with the sign flipped vectors as its rows.
	"""
	max_abs_rows = np.argmax(np.abs(u), axis=1)
	signs = np.sign(u[range(u.shape[0]), max_abs_rows])
	u *= signs[:, np.newaxis]
	return u
