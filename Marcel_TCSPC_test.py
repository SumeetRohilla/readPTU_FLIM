# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:30:29 2024

@author: narai
"""


import numpy as np
import os, pickle
import matplotlib.pyplot as plt
# from plantseg.predictions.functional.predictions import unet_predictions
# from sklearn.model_selection import ParameterGrid
# from tqdm import tqdm
# from plantseg.segmentation.functional.segmentation import mutex_ws
from PTU_ScanRead import Harp_TCPSC
# import glob
from FLIM_fitter import Calc_mIRF, FluoFit, DistFluoFit


#%%

filename = r'D:\Collabs\fromMarcel\AKT.ptu'

# head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead(filename) # Reading photon stream

tcspcdata, binT, head = Harp_TCPSC(filename)  
resolution = head['MeasDesc_Resolution'] * 1e9
tcspcIRF = Calc_mIRF(head, np.transpose(tcspcdata)[:,:,np.newaxis])
     

for ch in range(tcspcdata.shape[1]):
    IRF = np.squeeze(tcspcIRF[ch, :, 0])
    tmpi = np.where((IRF/np.max(IRF))<(10**-4))[0]
    IRF[tmpi]=0
    tau0 = np.array([0.5, 2.0]) # initial guesses   
    taufit, A, _, _, _, _, _, _, _ = FluoFit(IRF, np.squeeze(tcspcdata[:,ch]), \
                                             np.floor(head['MeasDesc_GlobalResolution']*10**9 + 0.5), \
                                             resolution, tau0 )
        
    # taufit, A, _, _, _, _, _, _, _ = FluoFit(np.squeeze(tcspcIRF[ch,:,0]), np.squeeze(tcspcdata[:,ch]), \
    #                                          np.floor(head['MeasDesc_GlobalResolution']*10**9 + 0.5), \
    #                                          resolution, tau0 )
        
    cx, tau, _, _, _, _, _  = DistFluoFit( np.squeeze(tcspcdata[:,ch]), np.floor(head['MeasDesc_GlobalResolution']*10**9 + 0.5),\
                      resolution, irf=IRF, shift=(-10,10), flag=1, bild = True, N = 100, scattering = True)

# only channel 1 - all tcspc - two component fitting


    


