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


def crosscorrelation(dir_mag, pixelsizes, masks, xps, yps, NameOfData):
    
    i = 0
    maskc_s=[]
    for mask in masks: #  both images are now of the same size and the same no. of frames
        #xps 150, 288,288 while in matlab xp 324*336*50 (x,y,no.of frames)
        #yps 150, 288, 288
        x = np.arange(-((xps[0].shape[2]*2)-1)/2, (((xps[0].shape[2]*2)-1)/2))
        y = np.arange(-((xps[0].shape[1]*2)-1)/2, (((xps[0].shape[1]*2)-1)/2))
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2+Y**2) #convert to polar coordinate axis
        maximumOfRho = int(np.max(rho))
        lags = np.linspace(0,maximumOfRho, maximumOfRho+1) #its size needs to be changed to be 1*maximum(467)
        #R= np.zeros(shape=(1, xp.shape[0]), dtype=float)
        #mask = mask.astype(np.float) # to change the mask from logical to float
        mask_new_size_x = (mask.shape[0]*2)-1
        mask_new_size_y = (mask.shape[1]*2)-1
        temp_mask = im.fromarray(mask)
        new_resized_mask = temp_mask.resize((mask_new_size_x,mask_new_size_y),im.BICUBIC)
        
        new_resized_mask.convert('RGB').save('temporaryImage_%s.png'%NameOfData[i])
        img2 = cv2.imread('temporaryImage_%s.png' %NameOfData[i])
        # cv2.imshow('temporaryImage',img2)
        # cv2.waitKey(0)  
        # cv2.destroyAllWindows()
        new_resized_mask_bin = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _,thresh1 = cv2.threshold(new_resized_mask_bin,127,255,cv2.THRESH_BINARY)
        
        # new_resized_mask_bin=np.transpose(thresh1) #0-255
        #The follwoing lines to solve the problem of black points inside the nucleus
        cv2.imwrite('temporaryImage2_%s.png' %NameOfData[i], new_resized_mask_bin)
        img3 = cv2.imread('temporaryImage2_%s.png' %NameOfData[i])
        for ii in range(img3.shape[0]):
            for jj in range(img3.shape[1]):
                if(img3[ii,jj,0] !=255):
                    img3[ii,jj,0]=0
        cnts, hierarchy= cv2.findContours(img3[:,:,0],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour=max(cnts, key=cv2.contourArea)
        cv2.drawContours(img3, [max_contour], -1, (255,255,255), -1)
        cv2.imshow("img_processed_%s" %NameOfData[i], img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        new_resized_mask_bin2 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        maskc,radius = innerCircle.innerCircle(new_resized_mask_bin2)
        # for ii in range(maskc.shape[0]):   
        #     for jj in range(maskc.shape[1]):    
        #         if(maskc[ii][jj] ==0):        
        #             maskc[ii][jj]= np.NaN
        maskc_s.append(maskc)
        for iii in range(mask.shape[0]):
            for jjj in range(mask.shape[1]):
                if(mask[iii][jjj] ==255):
                    mask[iii][jjj]=1
        
        i = i+1
    
    
    # #This block for Testing and should be deleted in line 52 in atocorrelation
    # imgg = cv2.imread("MaskC_outputFrominnerCircle.png")
    # maskc= imgg[:,:,2].astype(np.float64)
    # cv2.imshow('maskc',maskc)
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()
        
        
    R=[]
    lengthR=[]
    maskc=maskc_s[0]
    if len(maskc_s) == 2:
        maskc = np.multiply(maskc_s[0],maskc_s[1])
        
    for ii in range(maskc.shape[0]):   
            for jj in range(maskc.shape[1]):    
                if(maskc[ii][jj] ==0):        
                    maskc[ii][jj]= np.NaN
    
    
    

    #maskc = new_resized_mask_bin

    #loop on all lags
    numberOfFrames = xps[0].shape[0]
    if xps[1].shape[0] < xps[0].shape[0]:
        numberOfFrames = xps[1].shape[0]  
    # xp_0=xps[0]
    # yp_0=yps[0]
    
    # xp_1=xps[1]
    # yp_1=yps[1]
    
    for lag in range(1,numberOfFrames):
        sub_R=[]
        args=[]
        if dir_mag == 'mag':
            print("Autocorrelation in magnitude:", (lag/(numberOfFrames-1))*100, "%")
            #calculate magnitude
            for i in (len(xps)):
                args.append(np.power(np.square(xps[i][(lag):] - xps[i][0:(numberOfFrames-lag)]) + np.square(yps[i][(lag):] - yps[i][0:(numberOfFrames-lag)]),0.5))
        elif dir_mag == 'dir':
            print("Autocorrelation in direction:", (lag/(numberOfFrames-1))*100, "%") 
            #slope= (yp[(lag):] - yp[0:(numberOfFrames-lag)])/(xp[(lag):] - xp[0:(numberOfFrames-lag)])
            #arg = np.arctan(slope)
            for i in (len(xps)):
                args.append( np.arctan2((yps[i][(lag):] - yps[i][0:(numberOfFrames-lag)]),(xps[i][(lag):] - xps[i][0:(numberOfFrames-lag)])))
        C = np.zeros(shape=(args[0].shape[0],(xps[0].shape[1]*2)-1,(xps[0].shape[2]*2)-1))
        for k in range(0,args[0].shape[0]):
            Z=[]
            if dir_mag == 'mag':
                  #apply mask
                  for i in (len(args)):
                      arg = args[i]
                      z=np.multiply(arg[k] , masks[i])
                      z[z==0] = np.nan
                      zm = np.nanmean(z)
                      z = z-zm
                      z[np.isnan(z)] = 0
                      Z.append(z)
            elif dir_mag == 'dir':
                for i in (len(args)):
                    arg = args[i]
                    z=np.multiply(np.exp(arg[k]*1j), masks[i]) #324*336 matlab
                    Z.append(z)
            #cross correlation 
            #Cross-correlation is a basic signal processing method, which is used to analyze the similarity 
            #between two signals with different lags. Not only can you get an idea of how well the two signals 
            #match with each other, but you also get the point of time or an index, 
            #where they are the most similar
            #Calculate the cross correlation and normalize by the energy of the argument
            if len(Z) == 2:
                denominator1= np.reshape(np.abs(Z[0]), (-1,1))
                denominator2= np.reshape(np.abs(Z[1]), (-1,1))
                C[k] = scipy.signal.correlate(Z[0], Z[1])/(np.sum(np.power(denominator1,2))+np.sum(np.power(denominator2,2)))
            else:
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
            if len(pixelsizes) == 2:
                pixelsize = (pixelsizes[0]+pixelsizes[1])/2
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
    for mask in masks:
        for ii in range(mask.shape[0]):
            for jj in range(mask.shape[1]):
                if(mask[ii][jj] ==1):
                    mask[ii][jj]=255
                  
    return R, lags