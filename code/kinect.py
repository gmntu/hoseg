import os
import cv2
import sys
import torch
import ctypes
import _ctypes
import numpy as np
import torch.nn as nn
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from model import FCN_ADF, FCN_NEW

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

class KinectRuntime(object):
    def __init__(self):

        self._done = False

        #############################
        ### Kinect runtime object ###
        #############################
        self.kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Depth)
        self.depth_width, self.depth_height = self.kinect.depth_frame_desc.Width, self.kinect.depth_frame_desc.Height # Default: 512, 424
        self.color_width, self.color_height = self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height # Default: 1920, 1080

        ########################
        ### Load the network ###
        ########################
        self.net = FCN_NEW(n=8, f=40)
        self.net.load_state_dict(torch.load('model_FCN_NEW.pkl'))  
        # self.net.load_state_dict(torch.load('model_FCN_ADF.pkl'))  
        self.net.eval()

        ###############
        ### Add GPU ###
        ###############
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)  
        self.softmax = nn.Softmax2d()

    def run(self):
        while not self._done:                   
            ##############################
            ### Get images from camera ###
            ##############################
            if self.kinect.has_new_depth_frame() and self.kinect.has_new_color_frame():
                depth_frame = self.kinect.get_last_depth_frame()
                depth_img = depth_frame.reshape(((self.depth_height, self.depth_width))).astype(np.int32) # Need to reshape from 1D to 2D
                color_frame = self.kinect.get_last_color_frame()
                color_img = color_frame.reshape(((self.color_height, self.color_width, 4))).astype(np.uint8) # Need to reshape from 1D to 2D
        
                ############################
                ### Align color to depth ###
                ############################
                CSP_Count = self.kinect._depth_frame_data_capacity 
                CSP_type = _ColorSpacePoint * CSP_Count.value
                CSP = ctypes.cast(CSP_type(), ctypes.POINTER(_ColorSpacePoint))
                self.kinect._mapper.MapDepthFrameToColorSpace(self.kinect._depth_frame_data_capacity,self.kinect._depth_frame_data, CSP_Count, CSP)   
                align_color_img = np.zeros((self.depth_height,self.depth_width, 4), dtype=np.uint8)
                colorXYs = np.copy(np.ctypeslib.as_array(CSP, shape=(512*424,))) # Convert ctype pointer to array
                colorXYs = colorXYs.view(np.float32).reshape(colorXYs.shape + (-1,)) # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
                colorXYs += 0.5
                colorXYs = colorXYs.reshape(424,512,2).astype(np.int)
                colorXs = np.clip(colorXYs[:,:,0], 0, self.color_width-1)
                colorYs = np.clip(colorXYs[:,:,1], 0, self.color_height-1)
                align_color_img[:, :] = color_img[colorYs, colorXs, :]                

                #################################
                ### Crop images to 400 by 400 ###
                #################################
                depth_img = depth_img[12:400+12, 56:400+56] # Crop 400 by 400
                align_color_img = align_color_img[12:400+12, 56:400+56] # Crop 400 by 400
                depth_img[depth_img>1500] = 0

                ##################################################
                ### Forward the input depth image into the FCN ###
                ##################################################
                depth = torch.from_numpy(depth_img)
                depth = depth.float().view(1,1,depth.size(0),depth.size(1)) # View as a batch of 1 
                depth = depth.to(self.device)
                output = self.net(depth) # output torch.FloatTensor torch.Size([1, 8, 400, 400])
                # Convert to probabilities between 0 to 1
                output = self.softmax(output) # output torch.FloatTensor torch.Size([1, 8, 400, 400])
                values, output  = torch.max(output, 1) # output1 torch.LongTensor torch.Size([1, 400, 400]) tensor(0) tensor(7)
                output[values<0.8] = 0 # Threshold probabilities that are less than 0.8
                
                ##############################
                ### Colorize to the output ###
                ##############################
                output = output[0,:,:].cpu().numpy().astype(np.uint8)  
                output = np.stack((output,output,output), axis=2)
                output[np.where((output == 1).all(axis = 2))] = [255,255,0] # Cyan Foreground Note BGR
                output[np.where((output == 2).all(axis = 2))] = [255,0,0]   # Blue Left Hand 
                output[np.where((output == 3).all(axis = 2))] = [0,0,255]   # Red Right Hand
                output[np.where((output == 4).all(axis = 2))] = [255,0,255] # Magenta Left Arm
                output[np.where((output == 5).all(axis = 2))] = [0,255,255] # Yellow Right Arm
                output[np.where((output == 6).all(axis = 2))] = [0,255,0]   # Green Object 
                output[np.where((output == 7).all(axis = 2))] = [9,127,255] # Orange Table 
                
                ##########################
                ### Display the images ###
                ##########################
                cv2.imshow('output', output)
                depth_img = (depth_img/1500*255).astype(np.uint8)
                cv2.imshow('depth', depth_img)
                align_color_img = cv2.bitwise_and(align_color_img, align_color_img, mask=depth_img)
                cv2.imshow('color', align_color_img)

            # Get user input
            key = cv2.waitKey(1)
            if key==27: # Press esc to break the loop
                self._done = True

        # Close Kinect sensor and quit
        self.kinect.close()

__main__ = "KinectV2"
main = KinectRuntime();
main.run();
