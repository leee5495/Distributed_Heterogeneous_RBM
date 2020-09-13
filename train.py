# -*- coding: utf-8 -*-
import numpy as np

from module.test import Test
from module.dhrbm import DHRBM
from module.data_manager import DataManager

if __name__ == "__main__":
    datapath = "./data"
    modelpath = "./model"
    
    data = DataManager(datapath)
    test = Test(datapath)
    
    cluster_train_data = np.concatenate([data.rating_train, data.meta_train], axis=1)
    train_data = np.concatenate([data.rating_train_with_last, data.meta_train_with_last], axis=1)
    valid_data = np.concatenate([data.rating_valid, data.meta_valid], axis=1)
    cluster_train_data = np.concatenate([cluster_train_data, valid_data], axis=0)
    num_train = train_data.shape[0]
    num_valid = valid_data.shape[0]

    # hyperparameters
    num_visible = train_data.shape[1]
    num_hidden = 256
    k = 5
    num_cluster = 3
    batch_size = 20
    epochs = 100
    bootstrap_epochs = 50
    ensemble_epochs = 8
    dhrbm = DHRBM(num_visible, num_hidden, k, num_cluster)
    
    # train model
    dhrbm.train_cluster_rbm(cluster_train_data, epochs, bootstrap_epochs, batch_size)
    dhrbm.train_ensemble_model(cluster_train_data, train_data, num_train, ensemble_epochs, batch_size)
    dhrbm.save_model(modelpath)
    
    # validation
    prediction = dhrbm.predict(valid_data)
    hr_10 = test.hit_rate(prediction, data.last_rating_test, 10)
    hr_25 = test.hit_rate(prediction, data.last_rating_test, 25)
    arhr = test.arhr(prediction, data.last_rating_test)
    print("HR_10 : {:.02f}".format(hr_10))
    print("HR_25 : {:.02f}".format(hr_25))
    print("ARHR:   {:02f}".format(arhr))
    
    sample_ind = 100
    output_vec = prediction[sample_ind]
    input_vec = valid_data[sample_ind]
    test.explain_prediction(output_vec, input_vec)