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

#importing the training and testing data 
k=open(r"F:\datasets\cv_corpus_v1\dataset2500.plk","rb")
dataset=pickle.load(k)
X=[]
Y=[]
for i in dataset:
        X.append(i[0])
        Y.append(i[1])
     
#data=generator.train_test_generator(X,Y,100,3)
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.1,random_state=0)
#The model
model=keras.models.Sequential()
Conv2d=keras.layers.Conv2D
GRU=keras.layers.GRU
Dropout=keras.layers.Dropout
pooling=keras.layers.AveragePooling2D
l2=keras.regularizers.l2

model.add(Conv2d(64,[2,2],input_shape=[96,128,3],activation="relu",kernel_regularizer=l2(0.001)))

model.add(Dropout(0.1))
model.add(pooling(pool_size=(2,2)))
model.add(Conv2d(64,[2,2],activation="relu",kernel_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(2,2)))
model.add(Conv2d(64,[2,2],activation="relu"))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(3,3)))
model.add(Conv2d(32,[2,2],activation="relu"))
print("error1")
model.add(Dropout(0.1))

model.add(Conv2d(32,[6,1],activation="relu"))
print("error1")
model.add(Dropout(0.1))
model.add(keras.layers.Reshape([9,32]))
model.add(GRU(50))
#model.add(keras.layers.Reshape([150,1]))
#model.add(GRU())
model.add(keras.layers.Dense(200,activation="relu"))
model.add(Dropout(0.4))
model.add(keras.layers.Dense(3,activation="softmax"))
model.compile("Adadelta",loss='categorical_crossentropy',metrics=["accuracy"])
checkpoint=keras.callbacks.ModelCheckpoint("{epoch}-{val_acc}",monitor='val_acc',mode="auto",save_best_only=True)
earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
model.fit(np.array(X_train),np.array(Y_train),200,600,validation_data=[np.array(X_test),np.array(Y_test)],shuffle=True,callbacks=[earlystop,checkpoint])
model.save("modelConv5gru5b300e500d.h5")