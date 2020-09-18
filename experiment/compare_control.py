# -*- coding: utf-8 -*-
import os
import sys
import time

import torch
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

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
    dhrbm_prediction = dhrbm.predict(test_data)
    end_time = time.time()
    
    dhrbm_HR10 = test.hit_rate(dhrbm_prediction, last_item, 10)
    dhrbm_HR25 = test.hit_rate(dhrbm_prediction, last_item, 25)
    dhrbm_arhr = test.arhr(dhrbm_prediction, last_item)
    dhrbm_time = end_time - start_time
    
    # control: rbm with num_hidden = 256
    train_data = np.concatenate([data.rating_train_with_last, data.meta_train_with_last], axis=1)
    valid_data = np.concatenate([data.rating_valid, data.meta_valid], axis=1)
    entire_data = np.concatenate([train_data, valid_data], axis=0)
    num_hidden = 256
    epochs = 100
    batch_size = 20
    
    control_256_rbm = RBM(num_visible, num_hidden, k)
    control_256_rbm.train(entire_data, epochs, batch_size)
    
    start_time = time.time()
    control_256_prediction = control_256_rbm.predict(test_tensor).numpy()
    end_time = time.time()
    
    control_256_HR10 = test.hit_rate(control_256_prediction, last_item, 10)
    control_256_HR25 = test.hit_rate(control_256_prediction, last_item, 25)
    control_256_arhr = test.arhr(control_256_prediction, last_item)
    control_256_time = end_time - start_time
    
    # control: rbm with num_hidden = 512
    num_hidden = 512
    epochs = 100
    batch_size = 20
    
    control_512_rbm = RBM(num_visible, num_hidden, k)
    control_512_rbm.train(entire_data, epochs, batch_size)
    
    start_time = time.time()
    control_512_prediction = control_512_rbm.predict(test_tensor).numpy()
    end_time = time.time()
    
    control_512_HR10 = test.hit_rate(control_512_prediction, last_item, 10)
    control_512_HR25 = test.hit_rate(control_512_prediction, last_item, 25)
    control_512_arhr = test.arhr(control_512_prediction, last_item)
    control_512_time = end_time - start_time
    
    # control: rbm with num_hidden = 1024
    num_hidden = 1024
    epochs = 100
    batch_size = 20
    
    control_1024_rbm = RBM(num_visible, num_hidden, k)
    control_1024_rbm.train(entire_data, epochs, batch_size)
    
    start_time = time.time()
    control_1024_prediction = control_1024_rbm.predict(test_tensor).numpy()
    end_time = time.time()
    
    control_1024_HR10 = test.hit_rate(control_1024_prediction, last_item, 10)
    control_1024_HR25 = test.hit_rate(control_1024_prediction, last_item, 25)
    control_1024_arhr = test.arhr(control_1024_prediction, last_item)
    control_1024_time = end_time - start_time

    # print tabulated result
    table = tabulate([['HR10', dhrbm_HR10, control_256_HR10, control_512_HR10, control_1024_HR10], 
                      ['HR25', dhrbm_HR25, control_256_HR25, control_512_HR25, control_1024_HR25],
                      ['ARHR', dhrbm_arhr, control_256_arhr, control_512_arhr, control_1024_arhr],
                      ['time', dhrbm_time, control_256_time, control_512_time, control_1024_time]], 
                     headers=['M-Ensemble', 'Control-256', 'Control-512', 'Control-1024'],
                     tablefmt='orgtbl')
    print(table)

    # graph HR on each k
    ensemble_HR = [test.hit_rate(dhrbm_prediction, last_item, i) for i in range(10)]
    control_256_HR = [test.hit_rate(control_256_prediction, last_item, i) for i in range(10)]
    control_512_HR = [test.hit_rate(control_512_prediction, last_item, i) for i in range(10)]
    control_1024_HR = [test.hit_rate(control_1024_prediction, last_item, i) for i in range(10)]
    
    plt.plot(ensemble_HR)
    plt.plot(control_256_HR)
    plt.plot(control_512_HR)
    plt.plot(control_1024_HR)
    plt.ylabel('Hit Rate')
    plt.xlabel('k')
    plt.legend(['M-Ensemble', 'Control-256', 'Control-512', 'Control-1024'])
    plt.show()