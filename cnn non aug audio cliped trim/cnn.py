# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:25:49 2018

@author: Akshay

"""

from tensorflow import keras
import pickle
from sklearn import cross_validation
import matplotlib.pyplot as plt

import numpy as np
# data
data=open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_cliped_padded_mfcc_dataset_test_train.plk","rb")
data=pickle.load(data)
X_train,X_test,Y_train,Y_test=data
#The model
model=keras.models.Sequential()
Conv2d=keras.layers.Conv2D
GRU=keras.layers.GRU
Dropout=keras.layers.Dropout
pooling=keras.layers.AveragePooling2D
l2=keras.regularizers.l2
flatten=keras.layers.Flatten
model.add(Conv2d(64,[4,4],input_shape=[13,2378,1],activation="relu",kernel_regularizer=l2(0.01)))

model.add(Dropout(0.2))
model.add(pooling(pool_size=(1,2)))
model.add(Conv2d(64,[3,3],activation="relu",kernel_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(1,2)))
model.add(Conv2d(64,[3,3],activation="relu",kernel_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(1,3)))
model.add(Conv2d(32,[2,2],activation="relu",kernel_regularizer=l2(0.01)))

model.add(Dropout(0.2))
model.add(pooling(pool_size=(2,5)))
model.add(Conv2d(32,[2,2],activation="relu"))

model.add(Dropout(0.1))
model.add(keras.layers.Reshape([18,32]))
model.add(flatten())
#model.add(keras.layers.Reshape([150,1]))
#model.add(GRU())
model.add(keras.layers.Dense(100,activation="relu"))
model.add(Dropout(0.4))
model.add(keras.layers.Dense(5,activation="softmax"))
model.compile("Adadelta",loss='categorical_crossentropy',metrics=["accuracy"])
checkpoint=keras.callbacks.ModelCheckpoint("{epoch}-{val_acc}.h5",monitor='val_acc',mode="auto",save_best_only=True)
#earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
model.fit(np.array(X_train).reshape([-1,13,2378,1]),np.array(Y_train),50,500,validation_data=[np.array(X_test).reshape([-1,13,2378,1]),np.array(Y_test)],shuffle=True,callbacks=[checkpoint])
model.save("modelConv5b50e500.h5")