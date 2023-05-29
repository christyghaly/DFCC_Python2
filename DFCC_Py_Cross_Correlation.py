# -*- coding: utf-8 -*-
"""
Created on Sat May 27 21:21:15 2023

@author: hp
"""

# System imports  
import argparse

# System packages 
import os
import numpy
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
import crosscorrelation
# from core import innerCircle
# from core import radavg
from AutoCorrelationFit import AutoCorrelationFit
from PlotParameters import PlotParameters


####################################################################################################
# @__main__
####################################################################################################
if __name__ == "__main__":
    
    video_sequence1 = '%s/../data/protocol/rna/rna.avi' % os.getcwd()
    output_directory1 = '%s/../output-protocol/' % os.getcwd()
    pixel_threshold1 = 150
    pixel_size1 = 0.088
    dt1 = 0.2
    
    video_sequence2 = '%s/../data/protocol/dna/dna.avi' % os.getcwd()
    output_directory2 = '%s/../output-protocol/' % os.getcwd()
    pixel_threshold2 = 120
    pixel_size2 = 0.09
    dt2 = 0.2

    video_sequences=[]
    output_directories=[]
    pixel_thresholds=[]
    pixel_sizes=[]
    dts=[]
    
    video_sequences=[video_sequence1,video_sequence2]
    output_directories=[output_directory1, output_directory2]
    pixel_thresholds=[pixel_threshold1,pixel_threshold2]
    pixel_sizes=[pixel_size1,pixel_size2]
    dts=[dt1,dt2]
    
    
    #output
    Xpositions=[]
    YPositions=[]
    masks=[]
    
    x_size=[]
    y_size=[]
    
    i=0
    
    for video_sequence in video_sequences:
        frames = video_processing.get_frames_list_from_video(video_path=video_sequence, verbose=True)
        x_size.append(frames[0].shape[0])
        y_size.append(frames[0].shape[1])

        
    for video_sequence in video_sequences:
        
        nameOfData = pathlib.Path(video_sequence).stem
        # Get the prefix, typically with the name of the video sequence
        prefix = '%s_pixel-%2.2f_dt-%2.2f_threshold_%s' % (pathlib.Path(video_sequence).stem, pixel_sizes[i], dts[i], pixel_thresholds[i])
        ################# PLEASE DON'T EDIT THIS PANEL #################
        # Verify the input parameters, and return the path where the output data will be written  
        output_directory = file_utils.veryify_input_options(
            video_sequence=video_sequence, output_directory=output_directories[i], 
            pixel_threshold=pixel_thresholds[i], pixel_size=pixel_sizes[i], dt=dts[i])
        
        
        # Load the frames from the video 
        frames = video_processing.get_frames_list_from_video(
            video_path=video_sequence, verbose=True)
        newFrames=[]
        
        if (x_size[0] > x_size[1]) and i == 0:
            diff = (x_size[0]-x_size[1])
            start = diff/2
            end = x_size[0]-start
            for eachFrame in frames:
                newFrames.append(eachFrame[start:end, start:end])
        elif (x_size[1]> x_size[0]) and i == 1:
            for eachFrame in frames:
                newFrames.append(eachFrame[8:296, 23:311]) #it is done so as dna is not centralized 
            # diff = (x_size[1]-x_size[0])
            # start = diff/2
            # end = x_size[1]-start
            # for eachFrame in frames:
            #     newFrames.append(eachFrame[start:end, start:end])
            
            
        
        # Plot the first frames
        plotting.verify_plotting_packages()
        plotting.plot_frame(frame=newFrames[0], output_directory=output_directory, 
                            frame_prefix=prefix, font_size=14, tick_count=3)
        # Compute the optical flow
        print('* Computing optical flow') 
        u, v = optical_flow.compute_optical_flow_farneback(frames=newFrames)
        
        # Interpolate the flow field
        print('* Computing interpolations')
        u, v = optical_flow.interpolate_flow_fields(u_arrays=u, v_arrays=v)
        
        # Compute the trajectories 
        print('* Creating trajectories')
        trajectories = optical_flow.compute_trajectories(
            frame=newFrames[0], fu_arrays=u, fv_arrays=v, pixel_threshold=pixel_thresholds[i])
        
        number_frames=len(trajectories[0])
        XPos= numpy.zeros(shape=(number_frames,newFrames[0].shape[0], newFrames[0].shape[1]), dtype=float)
        YPos= numpy.zeros(shape=(number_frames,newFrames[0].shape[0], newFrames[0].shape[1]), dtype=float)
        for nonRefusedValues, trajectory in enumerate(trajectories):
            # ii = int(trajectory[0][0]) #26
            # jj =  int(trajectory[0][1]) #94
            ii = int(trajectory[0][1]) #94
            jj =  int(trajectory[0][0]) #26
            for kk in range(len(trajectory)):
                XPos[kk][ii][jj] = int(trajectory[kk][0])
                YPos[kk][ii][jj] = int(trajectory[kk][1])
                
        
        mask_matrix = numpy.zeros((newFrames[0].shape[0], newFrames[0].shape[1]), dtype=float)
        mask_matrix[numpy.where(newFrames[0] >= 60) ] = 255
        mask_matrix[numpy.where(newFrames[0] < 60) ] = 0
        filename = 'maskedImage.jpg'
        cv2.imwrite(filename, mask_matrix)
        
        temp_image = cv2.imread(filename)
        cv2.imshow('temp_mask',temp_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        Xpositions.append(XPos)
        YPositions.append(YPos)
        masks.append(mask_matrix)
        i = i + 1
    
    
    
    
    R,lags = crosscorrelation.crosscorrelation('dir', pixel_sizes,masks,Xpositions,YPositions)
    R_mag,_= crosscorrelation.crosscorrelation('mag',pixel_sizes,masks,Xpositions,YPositions)
    
    xi,nu = AutoCorrelationFit(lags,R)
    xi_mag,nu_mag = AutoCorrelationFit(lags,R_mag)
    PlotParameters(xi, nu, 0.2, xi_mag = xi_mag, nu_mag= nu_mag)
    print("finished")
    
    
    
    
