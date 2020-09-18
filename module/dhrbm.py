# -*- coding: utf-8 -*-
import os
import copy
import pickle

import torch
import numpy as np
from sklearn.cluster import KMeans

from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense

from module.rbm import RBM

class DHRBM:
    def __init__(self, num_visible, num_hidden, k, num_cluster, learning_rate=1e-2, momentum_coefficient=0.5, weight_decay=1e-4):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.num_cluster = num_cluster
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay

    def train_base_rbm(self, train_data, epochs, batch_size):
        self.base_rbm = RBM(self.num_visible, self.num_hidden, self.k, self.learning_rate, self.momentum_coefficient, self.weight_decay)
        self.base_rbm.train(train_data, epochs, batch_size)
        
    def cluster_data(self, train_data):
        train_set = torch.FloatTensor(train_data)
        base_hidden = self.base_rbm.sample_hidden(train_set).numpy()
        
        self.kmeans = KMeans(n_clusters=self.num_cluster).fit(base_hidden)
        labels = self.kmeans.labels_
        
        self.cluster_data = []
        for i in range(self.num_cluster):
            data_inds = np.where(labels == i)[0]
            data = train_data[data_inds]
            self.cluster_data.append(data)
            
    def train_cluster_rbm(self, train_data, epochs, bootstrap_epochs, batch_size):
        self.train_base_rbm(train_data, epochs, batch_size)
        self.cluster_data(train_data)
        self.cluster_rbms = []
        for i in range(self.num_cluster):
            cluster_rbm = copy.deepcopy(self.base_rbm)
            cluster_rbm.weights_momentum = torch.zeros(self.num_visible, self.num_hidden)
            cluster_rbm.visible_bias_momentum = torch.zeros(self.num_visible)
            cluster_rbm.hidden_bias_momentum = torch.zeros(self.num_hidden)
        
            cluster_rbm.train(self.cluster_data[i], bootstrap_epochs, batch_size)
            self.cluster_rbms.append(cluster_rbm)
            
    def make_ensemble_model(self, input_dim, hidden_dim, output_dim):
        input_layer = Input(shape=(input_dim,))
        hidden_layer = Dense(hidden_dim, activation='relu')(input_layer)
        output = Dense(output_dim, activation='sigmoid')(hidden_layer)
        self.ensemble_model = Model(inputs=input_layer, outputs=output)
        temp_weight = np.mean(np.array([self.cluster_rbms[i].weights.numpy().T for i in range(self.num_cluster)]), axis=0)
        temp_bias = np.mean(np.array([self.cluster_rbms[i].visible_bias.numpy().T for i in range(self.num_cluster)]), axis=0)
        self.ensemble_model.layers[2].set_weights([temp_weight, temp_bias])
        self.ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def make_ensemble_input(self, train_data, num_train):
        train_set = torch.FloatTensor(train_data)
        hiddens = []
        for i in range(self.num_cluster):
            hidden = self.cluster_rbms[i].sample_hidden(train_set).numpy()[:num_train]
            hiddens.append(hidden)
        base_hidden = self.base_rbm.sample_hidden(train_set).numpy()[:num_train]
        kmeans_dist = self.kmeans.transform(base_hidden)
        
        ensemble_input_data = np.concatenate(hiddens+[kmeans_dist], axis=1)
        return ensemble_input_data
        
    def train_ensemble_model(self, train_data, output_data, num_train, epochs, batch_size):
        ensemble_input = self.make_ensemble_input(train_data, num_train)
        input_dim = ensemble_input.shape[1]
        output_dim = output_data.shape[1]
        hidden_dim = int((input_dim-3)/3)
        
        self.make_ensemble_model(input_dim, hidden_dim, output_dim)
        history = self.ensemble_model.fit(ensemble_input, output_data,
                                          batch_size=batch_size, epochs=epochs,
                                          verbose=1)
        return history
    
    def predict(self, test_data):
        ensemble_input = self.make_ensemble_input(test_data, test_data.shape[0])
        return self.ensemble_model.predict(ensemble_input)
    
    def save_model(self, modelpath):
        self.base_rbm.save(os.path.join(modelpath, "base_rbm.pt"))
        for i in range(self.num_cluster):
            self.cluster_rbms[i].save(os.path.join(modelpath, "cluster_rbm{}.pt".format(i+1)))
        with open(os.path.join(modelpath, "kmeans"), "wb") as fp:
            pickle.dump(self.kmeans, fp)
        self.ensemble_model.save(os.path.join(modelpath, "ensemble_model"))
        
    def load_model(self, modelpath):
        self.base_rbm = RBM(self.num_visible, self.num_hidden, self.k, self.learning_rate, self.momentum_coefficient, self.weight_decay)
        self.base_rbm.load(os.path.join(modelpath, "base_rbm.pt"))
        
        self.cluster_rbms = []
        for i in range(self.num_cluster):
            self.cluster_rbms.append(RBM(self.num_visible, self.num_hidden, self.k, self.learning_rate, self.momentum_coefficient, self.weight_decay))
        for i in range(self.num_cluster):
            self.cluster_rbms[i].load(os.path.join(modelpath, "cluster_rbm{}.pt".format(i+1)))
            
        with open(os.path.join(modelpath, "kmeans"), "rb") as fp:
            self.kmeans = pickle.load(fp)
            
        self.ensemble_model = load_model(os.path.join(modelpath, "ensemble_model"))