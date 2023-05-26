# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 01:13:01 2023

@author: Christeen
"""
import numpy as np
from PIL import Image as im
# import innerCircle
import scipy.signal 
# import radavg
import cv2
# import core
from core import innerCircle
from core import radavg


def autocorrelation(dir_mag, pixelsize, mask, xp, yp):
    
    #xp 150, 288,288 while in matlab xp 324*336*50 (x,y,no.of frames)
    #yp 150, 288, 288
    x = np.arange(-((xp.shape[2]*2)-1)/2, (((xp.shape[2]*2)-1)/2))
    y = np.arange(-((xp.shape[1]*2)-1)/2, (((xp.shape[1]*2)-1)/2))
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2+Y**2) #convert to polar coordinate axis
    maximumOfRho = int(np.max(rho))
    lags = np.linspace(0,maximumOfRho, maximumOfRho+1) #its size needs to be changed to be 1*maximum(467)
    #R= np.zeros(shape=(1, xp.shape[0]), dtype=float)
    R=[]
    lengthR=[]
    
    
    #mask = mask.astype(np.float) # to change the mask from logical to float
    mask_new_size_x = (mask.shape[0]*2)-1
    mask_new_size_y = (mask.shape[1]*2)-1
    temp_mask = im.fromarray(mask)
    new_resized_mask = temp_mask.resize((mask_new_size_x,mask_new_size_y),im.BICUBIC)
    

    
    new_resized_mask.convert('RGB').save('temporaryImage.png')

    img2 = cv2.imread('temporaryImage.png')
    cv2.imshow('temporaryImage',img2)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
  
    new_resized_mask_bin = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _,thresh1 = cv2.threshold(new_resized_mask_bin,127,255,cv2.THRESH_BINARY)
    # new_resized_mask_bin=np.transpose(thresh1) #0-255
    #The follwoing lines to solve the problem of black points inside the nucleus
    
    cv2.imwrite('temporaryImage2.png', new_resized_mask_bin)
    img3 = cv2.imread('temporaryImage2.png')
    for ii in range(img3.shape[0]):
        for jj in range(img3.shape[1]):
            if(img3[ii,jj,0] !=255):
                img3[ii,jj,0]=0
    cnts, hierarchy= cv2.findContours(img3[:,:,0],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour=max(cnts, key=cv2.contourArea)
    cv2.drawContours(img3, [max_contour], -1, (255,255,255), -1)
    cv2.imshow("img_processed", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_resized_mask_bin2 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    
    
    maskc,radius = innerCircle.innerCircle(new_resized_mask_bin2)
    
    
    # #This block for Testing and should be deleted in line 52 in atocorrelation
    # imgg = cv2.imread("MaskC_outputFrominnerCircle.png")
    # maskc= imgg[:,:,2].astype(np.float64)
    # cv2.imshow('maskc',maskc)
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()

    #maskc = new_resized_mask_bin
    for i in range(maskc.shape[0]):   
        for j in range(maskc.shape[1]):    
            if(maskc[i][j] ==0):        
                maskc[i][j]= np.NaN
                
    for ii in range(mask.shape[0]):
        for jj in range(mask.shape[1]):
            if(mask[ii][jj] ==255):
                mask[ii][jj]=1
    #loop on all lags
    numberOfFrames = xp.shape[0]           
    for lag in range(1,numberOfFrames):
        sub_R=[]
        if dir_mag == 'mag':
            print("Autocorrelation in magnitude:", (lag/(numberOfFrames-1))*100, "%")
            #calculate magnitude
            arg = np.power(np.square(xp[(lag):] - xp[0:(numberOfFrames-lag)]) + np.square(yp[(lag):] - yp[0:(numberOfFrames-lag)]),0.5)
        elif dir_mag == 'dir':
           print("Autocorrelation in direction:", (lag/(numberOfFrames-1))*100, "%") 
           #slope= (yp[(lag):] - yp[0:(numberOfFrames-lag)])/(xp[(lag):] - xp[0:(numberOfFrames-lag)])
           #arg = np.arctan(slope)
           arg = np.arctan2((yp[(lag):] - yp[0:(numberOfFrames-lag)]),(xp[(lag):] - xp[0:(numberOfFrames-lag)]))
        C = np.zeros(shape=(arg.shape[0],(xp.shape[1]*2)-1,(xp.shape[2]*2)-1))
        for k in range(0,arg.shape[0]):
            if dir_mag == 'mag':
                 #apply mask
                 z=np.multiply(arg[k] , mask)
                 z[z==0] = np.nan
                 zm = np.nanmean(z)
                 z = z-zm
                 z[np.isnan(z)] = 0
            elif dir_mag == 'dir':
               z=np.multiply(np.exp(arg[k]*1j), mask) #324*336 matlab
            #cross correlation 
            #Cross-correlation is a basic signal processing method, which is used to analyze the similarity 
            #between two signals with different lags. Not only can you get an idea of how well the two signals 
            #match with each other, but you also get the point of time or an index, 
            #where they are the most similar
            #Calculate the cross correlation and normalize by the energy of the argument
            denominator= np.reshape(np.abs(z), (-1,1))
            C[k] = scipy.signal.correlate(z, z)/np.sum(np.power(denominator,2))
            #crop correlation function by rescaled version of circle shaped
            for i in range(maskc.shape[0]):   
                for j in range(maskc.shape[1]):    
                    if(maskc[i][j] == 255):        
                        maskc[i][j]= 1
            C[k]= np.multiply(C[k],maskc)
            
            #radial average
            C_real=np.real(C[k])
            out_radavg,out_lag=radavg.radavg(C_real,pixelsize) #shoukd return of size 1*260(non Nan values)
            if(k>1):
                if(out_radavg.size >= sub_R[k-1].size):
                    #sub_R[k] = out_radavg[0:sub_R[k-1].size]
                    sub_R.insert(k,out_radavg[0:sub_R[k-1].size])
                elif(out_radavg.size < sub_R[k-1].size):
                    
                    sub_R.insert(k,out_radavg)
            else:
                sub_R.insert(k,out_radavg)
        
        #for handling cutting out the values of the arrays to make them all of the same size
        #before : if sub_R = [array([ 5,  6, 11, 18]), array([ -1,  -2,   9,  19, 200, 201, 202])]
        #After the following loop sub_R = [array([ 5,  6, 11, 18]), array([-1, -2,  9, 19])]       
        length=[]
        for i in range(0,len(sub_R)):
            length.append(sub_R[i].size)
        mimum_length=min(length)
        for i in range(0,len(sub_R)):
            sub_R[i]=sub_R[i][0:mimum_length]
        
        lengthR.append(mimum_length)    
        R.insert(lag-1,sub_R)
    lengthMin=min(lengthR)
    
    #cropping all the lengths of the arrays to be all of the same length
    for i in range(0,len(R)):
        for j in range (0,len(R[i])):
            R[i][j]=R[i][j][0:lengthMin]
            
    
    lags= lags[0:lengthMin]
                  
    return R, lags