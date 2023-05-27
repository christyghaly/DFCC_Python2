# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:35:22 2023

@author: hp
"""

import cv2
import numpy as np

def innerCircle(mask):
    
    middle= (mask.shape[0]/2,mask.shape[1]/2) # find middle of the picture
    #getBoundaries
    #img = cv2.imread('mask_image.png')
    #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # convert to grayscale
    
    #we don't care about the hierarchial level beacuse it is not expected to be a child in the nucleus
    #cnts, hierarchy= cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #in order to get all the boundaries points we can uncomment the previous line which uses Chain appro
    # simple to decrease the boundaries line for memory optimization

            


    
    cnts, hierarchy= cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    # #The follwoing lines to solve the problem of black points inside the nucleus
    # copied_mask = mask.copy()
    # cv2.imshow('Mask3',mask)
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()
    # for ii in range(copied_mask.shape[0]):
    #     for jj in range(copied_mask.shape[1]):
    #         if(copied_mask[ii][jj] !=255):
    #             copied_mask[ii][jj]=0
    # cnts, hierarchy= cv2.findContours(copied_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # max_contour=max(copied_mask, key=cv2.contourArea)
    # cv2.drawContours(copied_mask, [max_contour], -1, (255,255,255), -1)
    # cv2.imshow("img_processed", copied_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    bound=np.zeros((2,2))
    for contour in cnts:
        bound = np.reshape(contour, (-1, 2)) # to change the tuple to size (numberOfPoints*2)
    
    #find minimum distance from midpoint to boundary
    arr1= [middle[0]]*bound.shape[0]
    arr2= [middle[1]]*bound.shape[0]
    C=np.vstack((arr1, arr2)).T
    sub = np.subtract(bound, C)
    powered_sub=np.power(sub, 2)
    res=np.sum(powered_sub, axis=1) # summing each row
    radius = min(np.power(res, 0.5))
    
    #take the values which are around the centroid and closer than the radius
    x, y = np.meshgrid(np.arange(1, mask.shape[1]+1), np.arange(1, mask.shape[0]+1))
    r= np.power(np.power((y-middle[0]),2)+np.power((x-middle[1]),2),0.5)
    maskc=np.empty(mask.shape, dtype=float) # was 647*671
    for i in range(mask.shape[0]):   
        for j in range(mask.shape[1]):    
            if(r[i][j] <= radius):        
                maskc[i][j]=1
            else:
               maskc[i][j]=0
    
    
    
    return maskc, radius