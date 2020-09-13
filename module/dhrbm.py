# -*- coding: utf-8 -*-
import os
import copy

import torch
import numpy as np
from sklearn.cluster import KMeans

from keras.models import Model
from keras.layers import Input, Dense

from rbm import RBM

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
            # train ensemble RBM1 on ensemble cluster 1
            cluster_rbm = copy.deepcopy(self.base_rbm)
            cluster_rbm.weights_momentum = torch.zeros(self.num_visible, self.num_hidden)
            cluster_rbm.visible_bias_momentum = torch.zeros(self.num_visible)
            cluster_rbm.hidden_bias_momentum = torch.zeros(self.num_hidden)
        
            cluster_rbm.train(self.cluster_data[i], epochs, batch_size)
            self.cluster_rbms.append(cluster_rbm)
            
    def make_ensemble_model(self, input_dim, hidden_dim, output_dim):
        input_layer = Input(shape=(input_dim,))
        hidden_layer = Dense(hidden_dim, activation='relu')(input_layer)
        output = Dense(output_dim, activation='sigmoid')(hidden_layer)
        self.ensemble_model = Model(inputs=input_layer, outputs=output)
        self.ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def make_ensemble_input(self, train_data, num_train):
        train_set = torch.FloatTensor(train_data)
        hiddens = []
        for i in range(self.num_cluster):
            hidden = self.cluster_rbms.sample_hidden(train_set).numpy()[:num_train]
            hiddens.append(hidden)
        base_hidden = self.base_rbm.sample_hidden(train_set).numpy()[:num_train]
        kmeans_dist = self.kmeans.transform(base_hidden)
        
        ensemble_input_data = np.concatenate(hiddens+[kmeans_dist], axis=1)
        return ensemble_input_data
    
    def make_ensemble_output(self, rating_data, meta_data):
        ensemble_output_data = np.concatenate([rating_data, meta_data], axis=1)
        return ensemble_output_data
        
    def train_ensemble_model(self, train_data, test_data, num_train, epochs, batch_size):
        ensemble_input = self.make_ensemble_input(train_data, num_train)
        ensemble_output = self.make_ensemble_output()
        
        input_dim = ensemble_input.shape[1]
        output_dim = ensemble_output.shape[1]
        hidden_dim = int((input_dim-3)/3)
        
        self.make_ensemble_model(input_dim, hidden_dim, output_dim)
        history = self.ensemble_model.fit(ensemble_input, ensemble_output,
                                          batch_size=batch_size, epochs=epochs,
                                          verbose=1)
        return history
    
    def predict(self, test_data):
        ensemble_input = self.make_ensemble_input(test_data, test_data.shape[0])
        self.ensemble_model.predict(ensemble_input)
        
            
        
            
    