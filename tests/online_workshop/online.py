#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:27:18 2018

@author: yangyc
"""

import numpy as np
from sklearn import preprocessing
import KindAP
global labels

data = np.array(final_data_object['no_identifiers_data_list'])
data = preprocessing.MinMaxScaler().fit_transform(data)
k = 3
model = KindAP.KindAP(n_clusters = k, algorithm = 'L', tol_in = 1e-5, 
                   max_iter_in = 1000)
clustering = model.fit(data)
labels = clustering.labels_
labels = labels.tolist()