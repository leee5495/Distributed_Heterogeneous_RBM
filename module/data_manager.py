# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np

class DataManager:
    def __init__(self, picklepath, use_valid=True, valid_rate=0.2, rand_seed=1):        
        self.picklepath = picklepath
        self.use_valid = use_valid
        if use_valid:
            self.valid_rate=0.2
        np.random.seed(rand_seed)
        self.open_datasets()
    
    def open_pickle(self, filename):
        with open(os.path.join(self.picklepath, filename), "rb") as fp:
            data = pickle.load(fp)
        return data
        
    def open_datasets(self):
        self.rating_train = np.array(self.open_pickle("rating_train"))
        self.meta_train = np.array(self.open_pickle("meta_train"))
        self.last_rating = np.array(self.open_pickle("last_rating"))
        self.last_meta = np.array(self.open_pickle("last_meta"))
        
        if self.use_valid:
            data_array = np.concatenate([self.rating_train, self.meta_train, self.last_rating, self.last_meta], axis=1)
            np.random.shuffle(data_array)

            rating_shape = self.rating_train.shape
            valid_ind = int(rating_shape[0]*self.valid_rate)
            self.rating_train = data_array[valid_ind:,:rating_shape[1]].astype(float)
            self.rating_valid = data_array[:valid_ind,:rating_shape[1]].astype(float)

            meta_shape = self.meta_train.shape
            self.meta_train = data_array[valid_ind:,rating_shape[1]:rating_shape[1]+meta_shape[1]].astype(float)
            self.meta_valid = data_array[:valid_ind,rating_shape[1]:rating_shape[1]+meta_shape[1]].astype(float)

            self.last_rating_train = data_array[valid_ind:, rating_shape[1]+meta_shape[1]:rating_shape[1]+meta_shape[1]+1].astype(int)
            self.rating_train_with_last = self.rating_train.copy()
            self.rating_train_with_last[np.arange(len(self.last_rating_train)), self.last_rating_train[:,0]] = 1
            self.last_rating_test = data_array[:valid_ind, rating_shape[1]+meta_shape[1]:rating_shape[1]+meta_shape[1]+1].astype(int)
            
            self.last_meta_train = data_array[valid_ind:, -3:]
            self.meta_train_with_last = self.meta_train.copy()
            num_genre = 164
            num_author = 2850
            for i in range(len(self.last_meta_train)):
                for j in self.last_meta_train[i,0]:
                    self.meta_train_with_last[i,j] = 1
                self.meta_train_with_last[i, num_genre+self.last_meta_train[i,1]] = 1
                self.meta_train_with_last[i, num_genre+num_author+self.last_meta_train[i,2]] = 1            
            self.last_meta_test = data_array[:valid_ind, -3:]