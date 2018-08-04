# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:22:51 2018

@author: Akshay
"""
# test set generator having equal classes
import numpy as np
import random
def train_test_generator(X,Y,n,num_class):
        x=X
        y=Y
        rand=[list(random.sample(range(100),n)) for i in range(num_class)]
        test_X=[]
        test_y=[]
        counter=[0]*num_class
        pop=[]
#        print(rand)
        for i in range(len(x)):
            class_no=np.argmax(y[i])
#            print(class_no)
            if len(rand[class_no])>0 and min(rand[class_no])==counter[class_no] :
                test_X.append(x[i])
                test_y.append(y[i])
                rand[class_no].remove(min(rand[class_no]))
                pop.append(i)
                
            counter[class_no]+=1    
        
        x=[x[i] for i in range(len(x)) if i not in pop]
        y=[y[i] for i in range(len(y)) if i not in pop]
        return x,test_X,y,test_y
           
        
            
    
