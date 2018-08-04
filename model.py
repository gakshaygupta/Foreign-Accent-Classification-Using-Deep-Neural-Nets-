# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:13:41 2018

@author: Akshay
"""

from tensorflow import keras
import pickle
from sklearn import cross_validation
import matplotlib.pyplot as plt

import numpy as np
#import generator
from imblearn.over_sampling import SMOTE, ADASYN
#importing the training and testing data 
k=open(r"F:\datasets\cv_corpus_v1\dataset1.plk","rb")
dataset=pickle.load(k)
X=[]
Y=[]
for i in dataset:
        X.append(i[0])
        Y.append(i[1])
     
#data=generator.train_test_generator(X,Y,100,3)
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.1)
#The model
model=keras.models.Sequential()
Conv2d=keras.layers.Conv2D
GRU=keras.layers.GRU
Dropout=keras.layers.Dropout
pooling=keras.layers.AveragePooling2D
model.add(Conv2d(32,[4,4],input_shape=[96,128,3],activation="relu"))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(2,2)))
model.add(Conv2d(64,[5,5],activation="relu"))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(3,3)))
model.add(Conv2d(32,[3,3],activation="relu"))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(2,2)))
model.add(Conv2d(32,[6,1],activation="relu"))
model.add(Dropout(0.1))
model.add(keras.layers.Reshape([8,32]))
model.add(GRU(100))
#model.add(keras.layers.Reshape([100,1]))
#model.add(GRU(50))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(3,activation="softmax"))
model.compile("Adagrad",loss='categorical_crossentropy',metrics=["accuracy"])
model.fit(np.array(X_train),np.array(Y_train),1,500,validation_data=[np.array(X_test),np.array(Y_test)],shuffle=True)
