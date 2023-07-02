# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:47:42 2023

@author: hp
"""
import numpy as np
from math import gamma
from scipy.special import kv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys

def func(x,b1,b2,b3):
    fourth_argumnet=kv(b3,np.divide(x,b2))
    third_arg=np.power(np.divide(x,b2),b3)
    second_arg=np.divide(2**(1-b3),gamma(b3))
    first_arg=np.multiply(b1, second_arg)
    functionToRet = np.multiply(np.multiply(first_arg,third_arg),fourth_argumnet)
    return functionToRet
    

def AutoCorrelationFit(lags, Correlation, magOrdir):
    indices = np.array(range(0,len(Correlation)))
    xi = np.zeros(shape=(len(indices),2))
    nu = np.zeros(shape=(len(indices),2))
    
    p_0 = np.array([1,3,3])
    
    
    if lags[0] == 0:
        lags[0] = sys.float_info.epsilon
        i = 0
    for i in range(len(indices)):
        y_notReshaped = np.mean(Correlation[i], axis=0)
        #xdata is lags
        #ydata is y which is the mean of the data
        y = np.reshape(y_notReshaped,(y_notReshaped.shape[1],))
        popt, pcov = curve_fit(func, lags, y,p0=p_0,check_finite=True, bounds=([sys.float_info.epsilon,0,0],[2000,10000,20]))
        coeffs_std = np.sqrt(np.diag(pcov))
        
        # plt.plot(lags,func(lags,popt[0],popt[1],popt[2]), c= colour[i])
        # plt.plot(lags,y, markerfacecolor=colour[i], marker="o", markeredgecolor=colour[i],markersize=1)
        # plt.xlabel('Spatial distance in pixels')
        # plt.ylabel('Correlation')
        # plt.savefig('plot%s_%ss.png'%(magOrdir,i))
        # plt.show()
        # plt.close()
        xi[i][0] = popt[1] # correlation length for a given lag
        xi[i][1] = coeffs_std[1] #correspomdimg standard deviation 
        nu[i][0] = popt[2] #smootheness parameter of a given lag
        nu[i][1] = coeffs_std[2] #corresponding std
        i = i+1
     
    return xi, nu