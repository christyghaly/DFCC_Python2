# -*- coding: utf-8 -*-
"""
Created on Sun May 28 00:17:14 2023

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
import autocorrelation
# from core import innerCircle
# from core import radavg
from AutoCorrelationFit import AutoCorrelationFit
from PlotParameters import PlotParameters


video_sequence2 = '%s/../data/protocol/dna/dna.avi' % os.getcwd()
output_directory2 = '%s/../output-protocol/' % os.getcwd()
pixel_threshold2 = 120
pixel_size2 = 0.09
dt2 = 0.2


nameOfData = pathlib.Path(video_sequence2).stem
prefix = '%s_pixel-%2.2f_dt-%2.2f_threshold_%s' % (pathlib.Path(video_sequence2).stem, pixel_size2, 0.2, 0.09)
        ################# PLEASE DON'T EDIT THIS PANEL #################
        # Verify the input parameters, and return the path where the output data will be written  
output_directory = file_utils.veryify_input_options(video_sequence=video_sequence2, output_directory=output_directory2, pixel_threshold=pixel_threshold2, pixel_size=pixel_size2, dt=dt2)
        
frames = video_processing.get_frames_list_from_video(video_path=video_sequence2, verbose=True)
mask_matrix = numpy.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=float)
mask_matrix[numpy.where(frames[0] >= 120) ] = 255
mask_matrix[numpy.where(frames[0] < 120) ] = 0
filename = 'maskedImage_dna.jpg'
cv2.imwrite(filename, mask_matrix)
cv2.imshow('mask', mask_matrix)

cv2.waitKey(0)  
cv2.destroyAllWindows()

crop_image = mask_matrix[8:296, 23:311]
cv2.imshow("Cropped", crop_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('ay7aga_%s.png'%nameOfData,crop_image)
