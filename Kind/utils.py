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
    """
    Get the confusion matrix
    :param clusters: predicted clustering labels
    :param classes_gt: real clustering labels
    :return: the confusion matrix
    """
    new_gt = deepcopy(classes_gt)
    unique_classes_gt = list(set(classes_gt))
    for i in range(len(classes_gt)):
        for j in range(len(unique_classes_gt)):
            if classes_gt[i] == unique_classes_gt[j]:
                new_gt[i] = j

    conf_mat = np.zeros([len(set(clusters)), len(set(new_gt))])
    for i in range(len(clusters)):
        conf_mat[clusters[i], new_gt[i]] += 1

    return conf_mat


def purity(clusters, classes_gt):
    """
    Get the purity based on confusion matrix
    :param clusters: predicted clustering labels
    :param classes_gt: real clustering labels
    :return:
    """
    conf_mat = confusion_matrix(clusters, classes_gt)
    sum_clu = np.max(conf_mat, axis=1)
    sum_tot = np.sum(sum_clu)
    pur = sum_tot / len(clusters)

    return pur


def proj_ud(nmat, data):
    """
    Project N to Ud
    :param nmat: R^{n*k}
    :param data: R^{n*d} usually d=k
    :return: P_{Ud} (N)
    """
    if sparse.issparse(nmat):
        smat, sigma, vmat = la.svd(sparse.csc_matrix.dot(data.T, nmat), full_matrices=False)
    else:
        smat, sigma, vmat = la.svd(np.dot(data.T, nmat), full_matrices=False)
    rotation = np.matmul(smat, vmat)
    U = np.matmul(data, rotation)
    return U


def proj_h(nmat, isnrm_col_H, isbinary_H):
    """
    Project N to H
    :param nmat: R^{n*k}
    :param isnrm_col_H: H's columns are normalized or not
    :param isbinary_H: H uses binary values or not
    :return: R^{n*k}
    """
    n, k = nmat.shape
    index = np.array(np.argmax(nmat, axis=1)).reshape((n,))
    if isbinary_H:
        values = np.array(np.max(nmat, axis=1)).reshape((n,))
    else:
        values = np.ones(n)
    H = sparse.csc_matrix((values, (np.arange(n), index)), shape=(n, k))
    if isnrm_col_H:
        H = normalize(H, axis=0)
    return H, index


def best_map(real_label, predicted_label):
    """
    find the best mapping between real labels and predicted clustering labels
    :param real_label:
    :param predicted_label:
    :return:
    """
    Label1 = np.unique(real_label)
    nClass1 = len(Label1)
    Label2 = np.unique(predicted_label)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = real_label == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = predicted_label == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(predicted_label.shape)
    for i in range(nClass2):
        newL2[predicted_label == Label2[i]] = Label1[c[i]]
    return newL2


def _deterministic_vector_sign_flip(u):
    """
    Modify the sign of vectors for reproducibility
    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.
    Parameters
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
