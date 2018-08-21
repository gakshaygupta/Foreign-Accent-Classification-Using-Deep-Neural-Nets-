# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 19:39:30 2018

@author: Akshay
"""

# data generator
import os 
import numpy as np
import pickle
import random
import re
def batch_gen(batch_size,folder,feature_shape,target_shape,c):
    l=os.listdir(folder)
    batch_features=np.zeros([batch_size,*feature_shape])
    batch_target=np.zeros([batch_size,*target_shape])
    coll=pickle.load(open(r"C:\Users\Akshay\Desktop\New folder (2)\sorted_no_of_data.plk","rb"))
    
    while True:
        if len(l)==0:
            l=os.listdir(folder)
        try:
            batch=random.sample(l,batch_size)
        except ValueError:
            batch=l.copy()
        l=list(set(l)-set(batch))
        print(batch)
        for i,j in enumerate(batch):
            alpha=open(r"{0}\{1}".format(folder,j),"rb")
            data=pickle.load(alpha)
            alpha.close()
            batch_features[i]=data[0].reshape(feature_shape)
            z=np.zeros(c)
            for k,g in enumerate(coll[:c]):
                if g[0]==re.findall(r"\D+",data[1])[0]:
                    z[k]=1    
                    print(z,k)
                    batch_target[i]=z
                    
        yield batch_features,batch_target
x       