# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:43:42 2023

@author: hp
"""

import pandas as pd
# import numpy as np
# import autocorrelation
# import cv2
# from AutoCorrelationFit import AutoCorrelationFit
# from PlotParameters import PlotParameters


import argparse

# System packages 
import os
import numpy as np
import pathlib 
import sys 
import warnings
import pickle
import cv2
warnings.filterwarnings('ignore') # Ignore all the warnings 

# Path hidpy
sys.path.append('%s/../' % os.getcwd())

# Internal packages 

import file_utils
import optical_flow
import video_processing
import plotting
import msd
# from core import inference
import autocorrelation
# from core import innerCircle
# from core import radavg
from AutoCorrelationFit import AutoCorrelationFit
from PlotParameters import PlotParameters

df = pd.read_excel(r'xp_1.xlsx')
arr1=df.to_numpy()
df2 = pd.read_excel(r'xp_2.xlsx')
arr2=df2.to_numpy()
df3 = pd.read_excel(r'xp_3.xlsx')
arr3=df3.to_numpy()
df4 = pd.read_excel(r'xp_4.xlsx')
arr4=df4.to_numpy()
df5 = pd.read_excel(r'xp_5.xlsx')
arr5=df5.to_numpy()
xp = np.zeros(shape=(5,arr1.shape[0],arr1.shape[1]))
xp[0]=arr1
xp[1]=arr2
xp[2]=arr3
xp[3]=arr4
xp[4]=arr5

df = pd.read_excel(r'yp_1.xlsx')
arr1=df.to_numpy()
df2 = pd.read_excel(r'yp_2.xlsx')
arr2=df2.to_numpy()
df3 = pd.read_excel(r'yp_3.xlsx')
arr3=df3.to_numpy()
df4 = pd.read_excel(r'yp_4.xlsx')
arr4=df4.to_numpy()
df5 = pd.read_excel(r'yp_5.xlsx')
arr5=df5.to_numpy()
yp = np.zeros(shape=(5,arr1.shape[0],arr1.shape[1]))
yp[0]=arr1
yp[1]=arr2
yp[2]=arr3
yp[3]=arr4
yp[4]=arr5

img = cv2.imread('mask_beforescaling.png')
print(type(img))
print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray',gray)
cv2.waitKey(0)  
cv2.destroyAllWindows()
R,lags = autocorrelation.autocorrelation('dir', 0.088,gray,xp,yp)

gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray2',gray2)
cv2.waitKey(0)  
cv2.destroyAllWindows()
R_mag,_= autocorrelation.autocorrelation('mag', 0.088,gray2,xp,yp)

xi,nu = AutoCorrelationFit(lags,R)
xi_mag, nu_mag = AutoCorrelationFit(lags, R_mag)



PlotParameters(xi, nu, 0.2, xi_mag = xi_mag, nu_mag= nu_mag)

print("Finished")
