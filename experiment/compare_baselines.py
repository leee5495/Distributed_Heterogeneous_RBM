# -*- coding: utf-8 -*-
import os
import sys
import time

import torch
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF

sys.path.append("..")
from module.test import Test
from module.dhrbm import DHRBM
from module.data_manager import DataManager

if __name__ == "__main__":
    modelpath = "../model"
    datapath = "../data"
    
    # import data
    data = DataManager(datapath)
    test_data = np.concatenate([data.rating_valid, data.meta_valid], axis=1)
    last_item = data.last_rating_test

    # open test module
    test = Test(datapath)
    
    # load trained dhrbm model
    num_visible = test_data.shape[1]
    num_hidden = 256
    k = 5
    num_cluster = 3
    dhrbm = DHRBM(num_visible, num_hidden, k, num_cluster)
    dhrbm.load_model(modelpath)
    
    # get dhrbm prediction
    start_time = time.time()
    dhrbm_prediction = dhrbm.predict(test_data)
    end_time = time.time()
    
    dhrbm_HR10 = test.hit_rate(dhrbm_prediction, last_item, 10)
    dhrbm_HR25 = test.hit_rate(dhrbm_prediction, last_item, 25)
    dhrbm_arhr = test.arhr(dhrbm_prediction, last_item)
    dhrbm_time = end_time - start_time
    
    # ItemPop prediction
    train_data = np.concatenate([data.rating_train, data.meta_train], axis=1)
    valid_data = np.concatenate([data.rating_valid, data.meta_valid], axis=1)
    entire_data = np.concatenate([train_data, valid_data], axis=0)
    
    start_time = time.time()
    itempop_prediction = np.sum(entire_data, axis=0, keepdims=True)
    itempop_prediction = np.tile(itempop_prediction, (len(last_item),1))
    end_time = time.time()
    
    itempop_HR10 = test.hit_rate(itempop_prediction, last_item, 10)
    itempop_HR25 = test.hit_rate(itempop_prediction, last_item, 25)
    itempop_arhr = test.arhr(itempop_prediction, last_item)
    itempop_time = end_time - start_time
    
    # ItemPop_Cluster
    start_time = time.time()
    base_rbm_hidden = dhrbm.base_rbm.sample_hidden(torch.FloatTensor(entire_data)).numpy()
    cluster_labels = dhrbm.kmeans.predict(base_rbm_hidden)
    cluster_pops = []
    for i in range(num_cluster):
        cluster_pop = np.sum(entire_data[np.where(cluster_labels==i)[0]], axis=0)
        cluster_pops.append(cluster_pop)
    itempop_cluster_prediction = []
    for i in cluster_labels[len(train_data):]:
        itempop_cluster_prediction.append(cluster_pops[i])
    itempop_cluster_prediction = np.array(itempop_cluster_prediction)
    end_time = time.time()
    
    itempop_cluster_HR10 = test.hit_rate(itempop_cluster_prediction, last_item, 10)
    itempop_cluster_HR25 = test.hit_rate(itempop_cluster_prediction, last_item, 25)
    itempop_cluster_arhr = test.arhr(itempop_cluster_prediction, last_item)
    itempop_cluster_time = end_time - start_time

    # SVD
    # de-mean (normalize by subtracting user mean) data
    user_mean = np.mean(entire_data, axis=1, keepdims=True)
    demeaned_input = entire_data - user_mean

    # SVD with k=10
    start_time = time.time()
    U_10, sigma_10, Vt_10 = svds(demeaned_input, k=10)
    sigma_10 = np.diag(sigma_10)
    svd_10_prediction = np.dot(np.dot(U_10, sigma_10), Vt_10) + user_mean
    end_time = time.time()
    
    svd_10_HR10 = test.hit_rate(svd_10_prediction[len(train_data):], last_item, 10)
    svd_10_HR25 = test.hit_rate(svd_10_prediction[len(train_data):], last_item, 25)
    svd_10_arhr = test.arhr(svd_10_prediction[len(train_data):], last_item)
    svd_10_time = end_time - start_time

    # SVD with k=50
    start_time = time.time()
    U_50, sigma_50, Vt_50 = svds(demeaned_input, k=50)
    sigma_50 = np.diag(sigma_50)
    svd_50_prediction = np.dot(np.dot(U_50, sigma_50), Vt_50) + user_mean
    end_time = time.time()
    
    svd_50_HR10 = test.hit_rate(svd_50_prediction[len(train_data):], last_item, 10)
    svd_50_HR25 = test.hit_rate(svd_50_prediction[len(train_data):], last_item, 25)
    svd_50_arhr = test.arhr(svd_50_prediction[len(train_data):], last_item)
    svd_50_time = end_time - start_time

    # NMF
    start_time = time.time()
    nmf = NMF(2)
    W = nmf.fit_transform(entire_data)
    H = nmf.components_
    nmf_prediction = np.dot(W, H)
    end_time = time.time()
    
    nmf_HR10 = test.hit_rate(nmf_prediction[len(train_data):], last_item, 10)
    nmf_HR25 = test.hit_rate(nmf_prediction[len(train_data):], last_item, 25)
    nmf_arhr = test.arhr(nmf_prediction[len(train_data):], last_item)
    nmf_time = end_time - start_time

    # print tabulated result
    table = tabulate([['HR10', dhrbm_HR10, itempop_HR10, itempop_cluster_HR10, svd_10_HR10, svd_50_HR10, nmf_HR10], 
                      ['HR25', dhrbm_HR25, itempop_HR25, itempop_cluster_HR25, svd_10_HR25, svd_50_HR25, nmf_HR25],
                      ['ARHR', dhrbm_arhr, itempop_arhr, itempop_cluster_arhr, svd_10_arhr, svd_50_arhr, nmf_arhr], 
                      ['time', dhrbm_time, itempop_time, itempop_cluster_time, svd_10_time, svd_50_time, nmf_time]],
                     headers=['Ensemble', 'ItemPop', 'ItemPop_Cluster', 'SVD10', 'SVD50', 'NMF'],
                     tablefmt='orgtbl')
    print(table)

    # graph HR on each k
    ensemble_HR = [test.hit_rate(dhrbm_prediction, last_item, i) for i in range(10)]
    itempop_HR = [test.hit_rate(itempop_prediction, last_item, i) for i in range(10)]
    itempop_cluster_HR = [test.hit_rate(itempop_cluster_prediction, last_item, i) for i in range(10)]
    svd_10_HR = [test.hit_rate(svd_10_prediction[len(train_data):], last_item, i) for i in range(10)]
    svd_50_HR = [test.hit_rate(svd_50_prediction[len(train_data):], last_item, i) for i in range(10)]
    nmf_HR = [test.hit_rate(nmf_prediction[len(train_data):], last_item, i) for i in range(10)]
    
    plt.plot(ensemble_HR)
    plt.plot(itempop_HR)
    plt.plot(itempop_cluster_HR)
    plt.plot(svd_10_HR)
    plt.plot(svd_50_HR)
    plt.plot(nmf_HR)
    plt.ylabel('Hit Rate')
    plt.xlabel('k')
    plt.legend(['M-Ensemble', 'ItemPop', 'ItemPop_Cluster', 'SVD10', 'SVD50', 'NMF'])
    plt.show()