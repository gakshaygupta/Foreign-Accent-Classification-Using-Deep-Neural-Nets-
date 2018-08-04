# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:01:54 2018

@author: Akshay
"""

import librosa
import os 
import matplotlib.pyplot as plt
import numpy as np
import librosa.display as display
import cv2
plt.ioff()
l=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\recordings")
sample,sr=librosa.core.load(r"C:\Users\Akshay\Desktop\New folder (2)\recordings\{0}".format(l[11]))
fig=plt.figure(figsize=(12, 8))

#d=librosa.feature.melspectrogram(sample)
logs=librosa.amplitude_to_db(librosa.stft(sample),ref=np.max)
display.specshow(logs, sr=sr, x_axis='time', y_axis='log')
#plt.imshow(logs)

plt.savefig(r"C:\Users\Akshay\Desktop\test.jpg")
