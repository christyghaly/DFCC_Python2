# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:39:07 2023

@author: hp
"""
import numpy as np




#Calculates radial average  of , pixel size 
    #Input:
        #array C of size 647*671 double with Nan values
        #pixelsize 0.088
    #output :
        #Radavg: radial average vector
        #lags: vector containg space lags
        #np.meshgrid(-C.shape[1]/2:(C.shape[1]/2)-1,-C.shape[0]/2:(C.shape[0]/2)-1)
        #X
def radavg(C,pixelSize):
    
    print("Entered")
    X,Y=np.meshgrid(np.arange(-C.shape[1]/2, (C.shape[1]/2)), np.arange(-C.shape[0]/2, C.shape[0]/2))
        #convert to polar coordinates
    rho = np.sqrt(X**2+Y**2) 
    rho=np.round(rho)
    maximumOfRho = int(np.max(rho))
    lags = np.linspace(0,maximumOfRho, maximumOfRho+1) #is of shape 467 and should be 1*467
    Radavg = np.full((1,int(lags.size)-1), np.NaN) #is of shape 1*466
        
    xc = (C.shape[0]+1)/2
    yc = (C.shape[1]+1)/2
        
    for i in range(1,lags.size):
        if((xc-lags[i] <1) or (yc-lags[i])<1):
            tempC=C
            tempRho=rho
        else:
            x_start = int(xc-lags[i])
            x_end = int(xc+lags[i])
            y_start = int(yc-lags[i])
            y_end = int(yc+lags[i])
                
            tempC= C[x_start-1:x_end,y_start-1:y_end]                
            tempRho= rho[x_start-1:x_end,y_start-1:y_end]
                
            logical_LHS = tempRho<=lags[i]
            logical_RHS = tempRho>lags[i-1]
            logical_arg = np.logical_and(logical_LHS,logical_RHS)
            Radavg[0][i]=np.nanmean(tempC[logical_arg])
                
                #if the number is NaN, break as we reached the border of the circle
            if(np.isnan(Radavg[0][i])):
                break
        
    Radavg=Radavg[np.logical_not(np.isnan(Radavg))] #excluding nan values
    Radavg=np.reshape(Radavg, (1, Radavg.size))
    Radavg[0][0]= C[int((C.shape[0]+1)/2)-1][int((C.shape[1]+1)/2)-1] 
    #Radavg[0]= C[(C.shape[0]+1)/2][(C.shape[1]+1)/2]
    lags = lags[0:(Radavg.size)]*pixelSize
    
    return Radavg,lags