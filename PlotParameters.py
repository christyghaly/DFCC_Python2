# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:56:00 2023

@author: hp
"""
import numpy as np

import matplotlib.pyplot as plt


def PlotParameters(xi_dir, nu_dir, dT,xi_mag=None, nu_mag=None):
    
    timeLag = np.array(range(1,xi_dir.shape[0]+1))*dT
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.errorbar(timeLag, xi_dir[:,0],yerr = xi_dir[:,1],fmt='-o',capsize=5, c= 'red')
    if xi_mag is not None:
        ax1.errorbar(timeLag, xi_mag[:,0],yerr = xi_mag[:,1],fmt='-o',capsize=5, c= 'green')
        
    ax1.set_xlabel('Time lag')
    ax1.set_ylabel('Correlation length')
    ax1.set_title('Line plot with error bars')
    
    ax2.errorbar(timeLag, nu_dir[:,0],yerr = nu_dir[:,1],fmt='-o',capsize=5, c= 'red')
    if nu_mag is not None:
        ax2.errorbar(timeLag, nu_mag[:,0],yerr = nu_mag[:,1],fmt='-o',capsize=5, c= 'green')
        
    
    ax2.set_xlabel('Time lag')
    ax2.set_ylabel('Smootheness Parameter')
    ax2.set_title('Line plot with error bars')
    plt.legend()
    plt.show()
    
    