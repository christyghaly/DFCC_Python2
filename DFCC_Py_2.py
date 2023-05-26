# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 04:07:56 2023

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



####################################################################################################
# @parse_command_line_arguments
####################################################################################################
def parse_command_line_arguments(arguments=None):
    """Parses the input arguments.
    :param arguments:
        Command line arguments.
    :return:
        Argument list.
    """

    description = 'hidpy is a pythonic implementation to the technique presented by Shaban et al, 2020.'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'The input configuration file that contains all the data. If this file is provided the other parameters are not considered'
    parser.add_argument('--config-file', action='store', help=arg_help, default='EMPTY')

    arg_help = 'The input stack or image sequence that will be processed to generate the output data'
    parser.add_argument('--input-sequence', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artifacts will be stored'
    parser.add_argument('--output-directory', action='store', help=arg_help)

    arg_help = 'The pixle threshold. (This value should be in microns, and should be known from the microscope camera)'
    parser.add_argument('--pixel-threshold', help=arg_help, type=float, default=10)

    arg_help = 'The pixle size. This value should be tested with trial-and-error'
    parser.add_argument('--pixel-size', help=arg_help, type=float)

    arg_help = 'Number of cores. If 0, it will use all the cores available in the system'
    parser.add_argument('--n-cores', help=arg_help, type=int, default=0)

    arg_help = 'Video time step.'
    parser.add_argument('--dt', help=arg_help, type=float)

    arg_help = 'Number of iterations, default 8'
    parser.add_argument('--iterations', help=arg_help, type=int, default=8)

    arg_help = 'Use the D model'
    parser.add_argument('--d-model', action='store_true')

    arg_help = 'Use the DA model'
    parser.add_argument('--da-model', action='store_true')

    arg_help = 'Use the V model'
    parser.add_argument('--v-model', action='store_true')

    arg_help = 'Use the DV model'
    parser.add_argument('--dv-model', action='store_true')
    
    arg_help = 'Use the DAV model'
    parser.add_argument('--dav-model', action='store_true')

    # Parse the arguments
    return parser.parse_args()


####################################################################################################
# @__main__
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

    if args.config_file == 'EMPTY':

        # video_sequence = args.input_sequence
        # output_directory = args.output_directory
        # pixel_threshold = args.pixel_threshold
        # pixel_size = args.pixel_size
        # video_sequence = '%s/../data/protocol/H2B_50Frames/U2OS_H2BGFP_example_data_50Frames.avi' % os.getcwd()
        # output_directory = '%s/../output-protocol/' % os.getcwd()
        # pixel_threshold = 150
        
        video_sequence = '%s/../data/protocol/rna/rna.avi' % os.getcwd()
        output_directory = '%s/../output-protocol/' % os.getcwd()
        pixel_threshold = 150
        pixel_size = 0.088

        dt = 0.2
        models_selected = list()
        models_selected.append('D')
        models_selected.append('DA')
        models_selected.append('V')
        models_selected.append('DV')
        models_selected.append('DAV')
        
    else:
        import configparser
        config_file = configparser.ConfigParser()

        # READ CONFIG FILE
        config_file.read(args.config_file)

        video_sequence = str(config_file['HID_PARAMETERS']['video_sequence'])
        output_directory = str(config_file['HID_PARAMETERS']['output_directory'])
        pixel_threshold = float(config_file['HID_PARAMETERS']['pixel_threshold'])
        pixel_size = float(config_file['HID_PARAMETERS']['pixel_size'])
        dt = float(config_file['HID_PARAMETERS']['dt'])
        ncores = int(config_file['HID_PARAMETERS']['n_cores'])
        
        d_model = config_file['HID_PARAMETERS']['d_model']
        da_model = config_file['HID_PARAMETERS']['da_model']
        v_model = config_file['HID_PARAMETERS']['v_model']
        dv_model = config_file['HID_PARAMETERS']['dv_model']
        dav_model = config_file['HID_PARAMETERS']['dav_model']
        
        models_selected = list()
        if d_model == 'Yes':
            models_selected.append('D')
        if da_model: 
            models_selected.append('DA')
        if v_model: 
            models_selected.append('V')
        if dv_model: 
            models_selected.append('DV')
        if dav_model: 
            models_selected.append('DAV')

    print(video_sequence)

    # Get the prefix, typically with the name of the video sequence  
    prefix = '%s_pixel-%2.2f_dt-%2.2f_threshold_%s' % (pathlib.Path(video_sequence).stem, pixel_size, dt, pixel_threshold)

    ################# PLEASE DON'T EDIT THIS PANEL #################
    # Verify the input parameters, and return the path where the output data will be written  
    output_directory = file_utils.veryify_input_options(
        video_sequence=video_sequence, output_directory=output_directory, 
        pixel_threshold=pixel_threshold, pixel_size=pixel_size, dt=dt)

    # Load the frames from the video 
    frames = video_processing.get_frames_list_from_video(
        video_path=video_sequence, verbose=True)


    # Plot the first frames
    plotting.verify_plotting_packages()
    plotting.plot_frame(frame=frames[0], output_directory=output_directory, 
        frame_prefix=prefix, font_size=14, tick_count=3)


    # Compute the optical flow
    print('* Computing optical flow') 
    u, v = optical_flow.compute_optical_flow_farneback(frames=frames)


    # Interpolate the flow field
    print('* Computing interpolations')
    u, v = optical_flow.interpolate_flow_fields(u_arrays=u, v_arrays=v)


    # Compute the trajectories 
    print('* Creating trajectories')
    trajectories = optical_flow.compute_trajectories(
        frame=frames[0], fu_arrays=u, fv_arrays=v, pixel_threshold=pixel_threshold)
    
    number_frames=len(trajectories[0])
    XPos= numpy.zeros(shape=(number_frames,frames[0].shape[0], frames[0].shape[1]), dtype=float)
    YPos= numpy.zeros(shape=(number_frames,frames[0].shape[0], frames[0].shape[1]), dtype=float)
    for nonRefusedValues, trajectory in enumerate(trajectories):
        # ii = int(trajectory[0][0]) #26
        # jj =  int(trajectory[0][1]) #94
        
        ii = int(trajectory[0][1]) #94
        jj =  int(trajectory[0][0]) #26
    
        for kk in range(len(trajectory)):
            XPos[kk][ii][jj] = int(trajectory[kk][0])
            YPos[kk][ii][jj] = int(trajectory[kk][1])


    print('* Creating the maps')

    mask_matrix = numpy.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=float)
    mask_matrix[numpy.where(frames[0] >= 60) ] = 255
    mask_matrix[numpy.where(frames[0] < 60) ] = 0
    
    cv2.imshow('Mask3',mask_matrix)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    
    filename = 'maskedImage.jpg'
    cv2.imwrite(filename, mask_matrix)
    
    
    R,lags = autocorrelation.autocorrelation('dir', 0.088,mask_matrix,XPos,YPos)
    R_mag,_= autocorrelation.autocorrelation('mag', 0.088,mask_matrix,XPos,YPos)
    
    xi,nu = AutoCorrelationFit(lags,R)
    xi_mag,nu_mag = AutoCorrelationFit(lags,R_mag)
    PlotParameters(xi, nu, 0.2, xi_mag = xi_mag, nu_mag= nu_mag)
    
    
    
    
