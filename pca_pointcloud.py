# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:21:19 2019

@author: sxd170023
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn import decomposition
from sklearn import datasets
import os
import argparse


"""
parser = argparse.ArgumentParser(description='PCA')
parser.add_argument('--path', type=str,default='D:/Sagnik/point_cloud/pristine')
args = parser.parse_args()
assert os.path.isdir(args.path)
"""
"""
c=0
files = []
dirpath = 'D:/Sagnik/point_cloud/pristine/'
for i in os.listdir(dirpath):
  img = os.path.join(dirpath,i)
  scan = np.fromfile(img,dtype=np.float32)
  X = scan.reshape((-1, 4))
  pca = decomposition.PCA(n_components=2)
  pca.fit(X)
  X1 = pca.transform(X)
  
  file_name = i.strip(".bin")
  np.savetxt(file_name,X1)
  c=c+1
print(c)
"""
fig = plt.figure()
X2= np.loadtxt('D:/Sagnik/point_cloud/pristine_pca/000100')
plt.plot(X2[:,0],X2[:,1])
plt.show()





img = 'D:/Sagnik/point_cloud/pristine/000100.bin'
scan = np.fromfile(img,dtype=np.float32)
X = scan.reshape((-1, 4))
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X1 = pca.transform(X)
#fig = plt.figure()
plt.plot(X1[:,0],X1[:,1])
plt.show()

