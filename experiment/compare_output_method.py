# -*- coding: utf-8 -*-
import os
import sys
import time
import copy

import torch
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.append("..")
from module.rbm import RBM
from module.test import Test
from module.dhrbm import DHRBM
from module.data_manager import DataManager

if __name__ == "__main__":
    modelpath = "../model"
    datapath = "../data"
    
    # import data
    data = DataManager(datapath)
    test_data = np.concatenate([data.rating_valid, data.meta_valid], axis=1)
    test_tensor = torch.FloatTensor(test_data)
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
    ensemble_prediction = dhrbm.predict(test_data)
    end_time = time.time()
    
    ensemble_HR10 = test.hit_rate(ensemble_prediction, last_item, 10)
    ensemble_HR25 = test.hit_rate(ensemble_prediction, last_item, 25)
    ensemble_arhr = test.arhr(ensemble_prediction, last_item)
    ensemble_time = end_time - start_time
    
    # train rbms for selection and weighted output methods
    epochs = 100
    bootstrap_epochs = 50
    batch_size = 20
    
    # train base rbm
    train_data = np.concatenate([data.rating_train_with_last, data.meta_train_with_last], axis=1)
    base_rbm = RBM(num_visible, num_hidden, k)
    base_rbm.train(train_data, epochs, batch_size)

    # get cluster rbm train input
    train_set = torch.FloatTensor(train_data)
    base_hidden = base_rbm.sample_hidden(train_set).numpy()
    kmeans = KMeans(n_clusters=num_cluster).fit(base_hidden)
    labels = kmeans.labels_
    cluster_data = []
    for i in range(num_cluster):
        data_inds = np.where(labels == i)[0]
        data = train_data[data_inds]
        cluster_data.append(data)
        
    # train cluster rbms
    cluster_rbms = []
    for i in range(num_cluster):
        cluster_rbm = copy.deepcopy(base_rbm)
        cluster_rbm.weights_momentum = torch.zeros(num_visible, num_hidden)
        cluster_rbm.visible_bias_momentum = torch.zeros(num_visible)
        cluster_rbm.hidden_bias_momentum = torch.zeros(num_hidden)
    
        cluster_rbm.train(cluster_data[i], bootstrap_epochs, batch_size)
        cluster_rbms.append(cluster_rbm)

    # get selection model prediction
    start_time = time.time()
    base_rbm_hidden = base_rbm.sample_hidden(test_tensor).numpy()
    cluster_rbm_predictions = []
    for i in range(num_cluster):
        cluster_rbm_predictions.append(cluster_rbms[i].predict(test_tensor).numpy())
    selection_prediction = []
    for i in range(len(test_tensor)):    
        best_model = kmeans.predict(base_rbm_hidden[i:i+1])[0] 
        selection_prediction.append(cluster_rbm_predictions[best_model][i])
    selection_prediction = np.array(selection_prediction)
    end_time = time.time()

    selection_HR10 = test.hit_rate(selection_prediction, last_item, 10)
    selection_HR25 = test.hit_rate(selection_prediction, last_item, 25)
    selection_arhr = test.arhr(selection_prediction, last_item)
    selection_time = end_time - start_time

    # get weighted model prediction
    start_time = time.time()
    base_rbm_hidden = base_rbm.sample_hidden(test_tensor).numpy()
    cluster_rbm_predictions = []
    for i in range(num_cluster):
        cluster_rbm_predictions.append(cluster_rbms[i].predict(test_tensor).numpy())
    weighted_prediction = []
    for i in range(len(test_tensor)):    
        best_model = kmeans.transform(base_rbm_hidden[i:i+1])[0] 
        model_scores = -best_model
        weights = np.reshape(np.exp(model_scores)/np.sum(np.exp(model_scores)), (-1,1))
        weighted_prediction.append(np.sum(np.array([cluster_rbm_prediction[i] for cluster_rbm_prediction in cluster_rbm_predictions])*weights, axis=0))
    weighted_prediction = np.array(weighted_prediction)
    end_time = time.time()

    weighted_HR10 = test.hit_rate(weighted_prediction, last_item, 10)
    weighted_HR25 = test.hit_rate(weighted_prediction, last_item, 25)
    weighted_arhr = test.arhr(weighted_prediction, last_item)
    weighted_time = end_time - start_time

    # print tabulated result
    table = tabulate([['HR10', selection_HR10, weighted_HR10, ensemble_HR10], 
                      ['HR25', selection_HR25, weighted_HR25, ensemble_HR25],
                      ['ARHR', selection_arhr, weighted_arhr, ensemble_arhr],
                      ['time', selection_time, weighted_time, ensemble_time]], 
                     headers=['M-Selection', 'M-Weighted', 'M-Ensemble'],
                     tablefmt='orgtbl')
    print(table)

    # graph HR on each k
    ensemble_HR = [test.hit_rate(ensemble_prediction, last_item, i) for i in range(10)]
    weighted_HR = [test.hit_rate(weighted_prediction, last_item, i) for i in range(10)]
    selection_HR = [test.hit_rate(selection_prediction, last_item, i) for i in range(10)]
    
    plt.plot(ensemble_HR)
    plt.plot(weighted_HR)
    plt.plot(selection_HR)
    plt.ylabel('Hit Rate')
    plt.xlabel('k')
    plt.legend(['M-Ensemble', 'M-Weighted', 'M-Selection'])
    plt.show()