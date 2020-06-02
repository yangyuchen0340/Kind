#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 19:43:05 2018

@author: yangyc

Run in ./matlab_data
"""

import time

# %% import libraries and data
import numpy as np
import pandas as pd
import scipy.io

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score, homogeneity_score
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score
import Kind

# %% Prepossessing
# Load data from .mat files
Uk = scipy.io.loadmat('Uk.mat')['Uk']
idxg = scipy.io.loadmat('idxg.mat')['idxg']
# pca = PCA(n_components=515)
# Data_t=pca.fit_transform(Data)
N, N_cluster = Uk.shape
# %% KindAP
# S,D,V=la.svd(Data,full_matrices=False)
print('--------------------------------')
t_start = time.time()
kindap = Kind.KindAP(n_clusters=N_cluster)
pred_kindAP = kindap.fit(Uk).labels_
t_end = time.time()
print('KindAP timing is ', t_end - t_start)
kindAP_accuracy1 = adjusted_mutual_info_score(idxg[:, 0], pred_kindAP)
kindAP_accuracy2 = v_measure_score(idxg[:, 0], pred_kindAP)
kindAP_accuracy3 = completeness_score(idxg[:, 0], pred_kindAP)
kindAP_accuracy4 = adjusted_rand_score(idxg[:, 0], pred_kindAP)
kindAP_accuracy5 = homogeneity_score(idxg[:, 0], pred_kindAP)
print('KindAP adjusted_mutual_info_score is ', kindAP_accuracy1)
print('KindAP v_measure_score is ', kindAP_accuracy2)
print('KindAP completeness_score is ', kindAP_accuracy3)
print('KindAP adjusted_rand_score is ', kindAP_accuracy4)
print('KindAP homogeneity_score is ', kindAP_accuracy5)
# %% K-means Clustering
print('--------------------------------')
t_start = time.time()
kmeans = KMeans(n_clusters=N_cluster, n_init=10, precompute_distances=False)
# kmeans.fit(Uk)
pred_kmeans = kmeans.fit_predict(Uk)
t_end = time.time()
print('K-means timing is ', t_end - t_start)
kmeans_accuracy1 = adjusted_mutual_info_score(idxg[:, 0], pred_kmeans)
kmeans_accuracy2 = v_measure_score(idxg[:, 0], pred_kmeans)
kmeans_accuracy3 = completeness_score(idxg[:, 0], pred_kmeans)
kmeans_accuracy4 = adjusted_rand_score(idxg[:, 0], pred_kmeans)
kmeans_accuracy5 = homogeneity_score(idxg[:, 0], pred_kmeans)
print('K-means adjusted_mutual_info_score is ', kmeans_accuracy1)
print('K-means v_measure_score is ', kmeans_accuracy2)
print('K-means completeness_score is ', kmeans_accuracy3)
print('K-means adjusted_rand_score is ', kmeans_accuracy4)
print('K-means homogeneity_score is ', kmeans_accuracy5)
df = pd.DataFrame(pred_kmeans)
df.to_csv('pred_kmeans.csv', index=None)
# %% Hierarchy Clustering
t_start = time.time()
hier = AgglomerativeClustering(n_clusters=N_cluster, linkage='ward')
pred_ward = hier.fit_predict(Uk)
t_end = time.time()
print('Ward timing is ', t_end - t_start)
ward_accuracy1 = adjusted_mutual_info_score(pred_ward, idxg[:, 0])
ward_accuracy2 = v_measure_score(pred_ward, idxg[:, 0])
ward_accuracy3 = completeness_score(idxg[:, 0], pred_ward)
ward_accuracy4 = adjusted_rand_score(idxg[:, 0], pred_ward)
ward_accuracy5 = homogeneity_score(idxg[:, 0], pred_ward)
print('Ward linkage adjusted_mutual_info_score is ', ward_accuracy1)
print('Ward linkage v_measure_score is ', ward_accuracy2)
print('Ward linkage completeness_score is ', ward_accuracy3)
print('Ward linkage adjusted_rand_score is ', ward_accuracy4)
print('Ward linkage homogeneity_score is ', ward_accuracy5)

# %% Cross Validation using MATLAB
pred_kmeans = np.squeeze(pd.read_csv('pred_kmeans_matlab.csv', header=None).values)
# The difference comes from different initialization in k-means++
# In matlab 2018a (kmeans.m line 428-445), k-means++ only samples once
# In python sklearn (sklearn/cluster/k_means_.py line 108-143), k-means++ samples n_local_trials
