# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 17:31:03 2018

@author: Akshay
"""
"""pading and triming code"""
import os
import numpy as np
import re
import collections 
import pickle
import librosa
import pylab
import librosa.display as display
import matplotlib
import cv2
matplotlib.use('Agg')

d=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram")

l=[re.findall(r"\D+",x)[0] for x in d]
#variabe coll contains the classes and their nunmber of instances
coll=collections.Counter(l)
coll=list(zip(list(coll.keys()),list(coll.values())))
coll=sorted(coll,key=lambda x:x[1],reverse=True)
with open(r"C:\Users\Akshay\Desktop\New folder (2)\sorted_no_of_data.plk",'wb') as o:
    pickle.dump(coll,o)
count=0
time=0
#average time calculation
for i in coll[0:3]:
    for  j in d:
        if i[0] in j:
            count+=1
            S,sr=librosa.load(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram\{0}".format(j))
            length=len(S)/sr
            time+=length
print("average length",time/count) 
#           
#with open(r"C:\Users\Akshay\Desktop\New folder (2)\avg.plk",'wb') as o:
#    pickle.dump(time/count,o)
    
k=open(r"C:\Users\Akshay\Desktop\New folder (2)\sorted_no_of_data.plk",'rb')    
#variabe coll contains the classes and their nunmber of instances
coll=pickle.load(k)
#code for creating spectrogram and saving it
for i in d:
    if coll[2][0]in i:
        
        sample,sr=librosa.core.load(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram\{0}".format(i))
        if len(sample)/sr>=30:
            sample=sample[:30*sr]
        else:
            S=np.zeros(30*sr)
            alpha=len(sample)
            S[:alpha]=sample
            sample=S
        #fig=plt.figure(figsize=(12, 8))
        #d=librosa.feature.melspectrogram(sample)
        logs=librosa.amplitude_to_db(librosa.stft(sample),ref=np.max)
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) 
        display.specshow(logs, sr=sr, x_axis='time', y_axis='log')
        pylab.savefig(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram2\{0}.jpg".format(i[:-4]), bbox_inches=None, pad_inches=0)
        pylab.close()  
 #resizing the spectrogram        
spec=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram2")        
for i in spec:
    image=cv2.imread(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram2\{0}".format(i))
    resized=cv2.resize(image,(128,96),interpolation=cv2.INTER_AREA)
    cv2.imwrite(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram3\{0}".format(i),resized)
#craeting dataset 
a=[]
k=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram3")
for i in k:
   onehot=np.zeros(3)
   if "arabic" in i:
      image=cv2.imread(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram3\{0}".format(i)) 
      onehot[0]=1
      a.append([image,onehot])
   elif "english" in i :
       image=cv2.imread(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram3\{0}".format(i)) 
       onehot[1]=1
       a.append([image,onehot])
   else:
      image=cv2.imread(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram3\{0}".format(i)) 
      onehot[2]=1
      a.append([image,onehot])
       
with open(r"C:\Users\Akshay\Desktop\New folder (2)\dataset.plk","wb") as o:
    pickle.dump(a,o)    
        
  