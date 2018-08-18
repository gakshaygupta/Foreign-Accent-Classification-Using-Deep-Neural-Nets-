# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:28:00 2018

@author: Akshay
"""

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
import generator

#importing the training and testing data 
k=open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_cliped_padded_mfcc_dataset.plk","rb")
dataset=pickle.load(k)
X=[]
Y=[]
for i in dataset:
        X.append(i[0])
        Y.append(i[1])
     
data=generator.train_test_generator(X,Y,10,5)
with open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_cliped_padded_mfcc_dataset_test_train.plk","wb") as a:
    pickle.dump(data,a)
    
X_train,X_test,Y_train,Y_test=data #cross_validation.train_test_split(X,Y,test_size=0.1,random_state=0)
#The model
model=keras.models.Sequential()
Conv2d=keras.layers.Conv2D
GRU=keras.layers.GRU
Dropout=keras.layers.Dropout
pooling=keras.layers.AveragePooling2D
l2=keras.regularizers.l2

model.add(Conv2d(64,[4,4],input_shape=[13,2378,1],activation="relu",kernel_regularizer=l2(0.01)))

model.add(Dropout(0.2))
model.add(pooling(pool_size=(1,2)))
model.add(Conv2d(64,[3,3],activation="relu",kernel_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(1,2)))
model.add(Conv2d(64,[2,2],activation="relu",kernel_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(pooling(pool_size=(1,3)))
model.add(Conv2d(32,[2,2],activation="relu",kernel_regularizer=l2(0.01)))

model.add(Dropout(0.2))
model.add(pooling(pool_size=(2,10)))
model.add(Conv2d(32,[3,3],activation="relu"))

model.add(Dropout(0.1))
model.add(keras.layers.Reshape([17,32]))
model.add(GRU(100))
#model.add(keras.layers.Reshape([150,1]))
#model.add(GRU())
model.add(keras.layers.Dense(100,activation="relu"))
model.add(Dropout(0.4))
model.add(keras.layers.Dense(5,activation="softmax"))
model.compile("Adadelta",loss='categorical_crossentropy',metrics=["accuracy"])
checkpoint=keras.callbacks.ModelCheckpoint("{epoch}-{val_acc}.h5",monitor='val_acc',mode="auto",save_best_only=True)
#earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
model.fit(np.array(X_train).reshape([-1,13,2378,1]),np.array(Y_train),50,500,validation_data=[np.array(X_test).reshape([-1,13,2378,1]),np.array(Y_test)],shuffle=True,callbacks=[checkpoint])
model.save("modelConv5gru100b100e500d3.h5")