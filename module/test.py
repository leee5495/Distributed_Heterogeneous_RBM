# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd

class Test:
    def __init__(self, datapath):
        with open(os.path.join(datapath, "book_id"), "r") as fin:
            self.book_id = [line.strip() for line in fin]
        self.ind_to_book_id = {idx: book_id for idx, book_id in enumerate(self.book_id)}
        with open(os.path.join(datapath, "genre_id"), "rb") as fin:
            self.genre_id = [line.strip() for line in fin]
        self.ind_to_genre = {idx: genre_id for idx, genre_id in enumerate(self.genre_id)}
        with open(os.path.join(datapath, "author_id"), "rb") as fin:
            self.author_id = [line.strip() for line in fin]
        self.ind_to_author = {idx: author_id for idx, author_id in enumerate(self.author_id)}
        with open(os.path.join(datapath, "publisher_id"), "rb") as fin:
            self.publisher_in = [line.strip() for line in fin]
        self.ind_to_publisher = {idx: publisher_id for idx, publisher_id in enumerate(self.publisher_id)}
        self.meta = pd.read_csv(os.path.join(datapath, "meta.csv"))
        
        self.num_books = len(self.book_id)
        self.num_genre = len(self.genre_id)
        self.num_author = len(self.author_id)
        self.num_publisher = len(self.publisher_id)
            
    def explain_prediction(self, output_vector, input_vector, top_n=5, use_meta=False, use_lda=False):
        rating_output = output_vector[:self.num_books]
        rating_input = input_vector[:self.num_books]
        
        # explain input
        print("Explain input")
        in_rating_ind = np.reshape(np.argwhere(rating_input>0), (-1,))
        for i in in_rating_ind:
            book_meta = self.meta[self.meta.book_id == i].iloc[0]
            print(" title:     ", book_meta.title)
            print(" author:    ", self.ind_to_author[book_meta.author])
            print(" genre:     ", [self.ind_to_genre[j] for j in book_meta.genre])
            print(" publisher: ", self.ind_to_publisher[book_meta.publisher])
            print(" rating:    ", rating_input[i]*5)
            print()
        
        # explain output
        print("\n\nExplain output")
        rating_output[in_rating_ind] = 0
        top_n_ind = (-rating_output).argsort()[:top_n]
        for i in top_n_ind:
            book_meta = self.meta[self.meta.book_id == i].iloc[0]
            print(" title:     ", book_meta.title)
            print(" author:    ", self.ind_to_author[book_meta.author])
            print(" genre:     ", [self.ind_to_genre[j] for j in book_meta.genre])
            print(" publisher: ", self.ind_to_publisher[book_meta.publisher])
            print(" score:     ", output_vector[i])
            print()
            
        if use_meta:
            print("\n\nExplain output meta")
            genre_output = output_vector[self.num_books:self.num_books+self.num_genre]
            author_output = output_vector[self.num_books+self.num_genre:self.num_books+self.num_genre+self.num_author]
            publisher_output = output_vector[self.num_books+self.num_genre+self.num_author:self.num_books+self.num_genre+self.num_author+self.num_publisher]
            
            print(" Genre: ")
            top_n_ind = (-genre_output).argsort()[:top_n]
            for i in top_n_ind:
                print("   -", self.ind_to_genre[i])

            print(" Author: ")
            top_n_ind = (-author_output).argsort()[:top_n]
            for i in top_n_ind:
                print("   -", self.ind_to_author[i])
                
            print(" Publisher: ")
            top_n_ind = (-publisher_output).argsort()[:top_n]
            for i in top_n_ind:
                print("   -", self.ind_to_publisher[i])

    def explain_prediction_batch(self, output_vectors, input_vectors, top_n=5, use_meta=False):
        for output_vector, input_vector in zip(output_vectors, input_vectors):
            self.explain_prediction(output_vector, input_vector, top_n=top_n, use_meta=use_meta)
            
    def hit_rate(self, test_output, next_books, k):
        rating_vect = test_output[:,:self.num_books]
        top_k = (-rating_vect).argsort()[:,:k]
        hit = 0
        total = len(rating_vect)
        for i in range(len(rating_vect)):
            if next_books[i][0] in top_k[i]:
                hit += 1
        return hit/total