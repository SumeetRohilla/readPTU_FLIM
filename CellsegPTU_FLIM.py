# -*- coding: utf-8 -*-
"""
CellsegPTU_FLIM.py - Cell Segmentation and FLIM Analysis Pipeline

This module provides a comprehensive pipeline for analyzing Fluorescence Lifetime Imaging
Microscopy (FLIM) data from PicoQuant PTU files. It combines advanced cell segmentation
using PlantSeg with fluorescence lifetime fitting to extract cellular fluorescence properties.

Key Functionalities:
- PTU file reading and photon stream processing
- Cell segmentation using U-Net deep learning models
- Parameter optimization for segmentation quality
- Fluorescence lifetime fitting (multi-exponential)
- Cellular FLIM analysis and visualization

Dependencies:
- numpy: Numerical computations and array operations
- matplotlib.pyplot: Plotting and visualization
- os, pickle, glob: File system operations and data serialization
- plantseg: Deep learning-based plant cell segmentation
- sklearn: Machine learning utilities (parameter grid search)
- tqdm: Progress bar visualization
- PTU_ScanRead: Custom module for PTU file reading and processing
- FLIM_fitter: Custom module for fluorescence lifetime fitting

Author: narain karedla
Created: Sun Sep  1 00:43:05 2024
"""

# Standard library imports
import numpy as np
import os
import pickle
import glob
from typing import Dict, List, Tuple, Optional, Any

# Third-party imports
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# PlantSeg imports for cell segmentation
from plantseg.predictions.functional.predictions import unet_predictions
from plantseg.segmentation.functional.segmentation import mutex_ws

# Custom module imports
from PTU_ScanRead import PTU_ScanRead, Process_Frame, mHist2
from FLIM_fitter import Calc_mIRF, FluoFit, DistFluoFit
# ============================================================================
# CONFIGURATION AND DATA LOADING SECTION
# ============================================================================

# File path configuration
filename = r'D:\Collabs\fromYuexuan\Yuexuan_ptu_testfile.ptu'  # Input PTU file path
res_file = filename[:-4] + '_FLIM_data.pkl'  # Cached processed data file

# Analysis parameters
cnum = 1  # Number of PIE (Pulsed Interleaved Excitation) cycles, default value
max_cell_Area = 5000  # Maximum cell area threshold in pixels for filtering

# Data loading: Check if preprocessed data exists to avoid reprocessing
if os.path.exists(res_file):
    """
    Load previously processed FLIM data from pickle file.
    This saves significant processing time for repeated analysis.
    """
    print(f"Loading cached FLIM data from: {res_file}")
    pklname = glob.glob(res_file)
    with open(pklname[0], 'rb') as f:
        FLIM_data = pickle.load(f)
    
    # Extract photon stream data arrays
    im_sync = FLIM_data['im_sync']      # Sync pulse timestamps
    im_tcspc = FLIM_data['im_tcspc']    # Time-correlated single photon counting data
    im_chan = FLIM_data['im_chan']      # Detector channel information
    im_line = FLIM_data['im_line']      # Line scan coordinates
    im_frame = FLIM_data['im_frame']    # Frame numbers
    im_col = FLIM_data['im_col']        # Column coordinates
    head = FLIM_data['head']            # Header information from PTU file
    # timeF = FLIM_data['time']         # Optional: timing information
    
    # Update global namespace with all loaded data
    globals().update(FLIM_data)
else:
    """
    Read and process raw PTU file data.
    This step extracts photon stream data from the binary PTU format.
    """
    print(f"Processing raw PTU file: {filename}")
    head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead(filename)
     
# ============================================================================
# PARAMETER SETUP AND INITIALIZATION
# ============================================================================

# TCSPC (Time-Correlated Single Photon Counting) parameters
resolution = 0.2  # ns - temporal resolution for TCSPC histogram binning
dind = np.unique(im_chan)  # Array of unique detector channel indices
nFrames = head['ImgHdr_MaxFrames']  # Total number of frames in the dataset

# PIE (Pulsed Interleaved Excitation) configuration
if 'PIENumPIEWindows' in head:
    cnum = head['PIENumPIEWindows']  # Number of PIE cycles from header
    print(f"PIE cycles detected: {cnum}")

# Optional parameters for advanced analysis (currently commented out)
# sync_pixel = head['ImgHdr_PixelTime'] * head['SyncRate'] * 1e-9  # syncs per pixel
# threshold = 0.1 * sync_pixel  # threshold per pixel for dead-time correction

# ============================================================================
# FLUORESCENCE CHANNEL CONFIGURATION
# ============================================================================

# Channel assignments for different fluorescence signals
# These parameters define which laser pulse and detector combinations
# correspond to specific cellular components

# Membrane stain channel configuration
mem_PIE = 2   # Laser pulse number for membrane channel (PIE window)
mem_det = 2   # Detector ID for membrane channel detection

# Autofluorescence channel configuration
auto_PIE = 1  # Laser pulse number for autofluorescence channel (PIE window)
auto_det = 1  # Detector ID for autofluorescence channel detection

print(f"Membrane channel: PIE={mem_PIE}, Detector={mem_det}")
print(f"Autofluorescence channel: PIE={auto_PIE}, Detector={auto_det}")

# ============================================================================
# SEGMENTATION PARAMETER OPTIMIZATION SETUP
# ============================================================================

# Parameter grid for systematic optimization of segmentation quality
# These parameters control the PlantSeg segmentation algorithm behavior
param_grid = {
    # Beta parameter for GASP (Graph-based Active Segmentation with Priors)
    # Lower values (0.5-0.7): tend towards under-segmentation (fewer, larger segments)
    # Higher values (0.8-0.95): tend towards over-segmentation (more, smaller segments)
    "beta": [round(x, 1) for x in np.arange(0.5, 0.95, 0.05)],
    
    # Minimum size threshold for post-processing
    # Segments smaller than this value will be merged or removed
    "post_minsize": [round(x, 1) for x in np.arange(190, 210, 10)],
}

# Generate all parameter combinations for grid search
params = list(ParameterGrid(param_grid))
print(f"Parameter combinations for optimization: {len(params)}")

# ============================================================================
# MAIN PROCESSING LOOP - FRAME-BY-FRAME ANALYSIS
# ============================================================================

print(f"Processing {nFrames} frames for cell segmentation and FLIM analysis...")

for nz in range(nFrames):
    print(f"\n--- Processing Frame {nz + 1}/{nFrames} ---")
    
    # Extract photon indices for current frame
    ind = np.where(im_frame == nz)[0]
    print(f"Frame {nz}: Processing {len(ind)} photons")
    
    # ========================================================================
    # PHOTON STREAM PROCESSING AND IMAGE RECONSTRUCTION
    # ========================================================================
    
    # Process photon stream data for current frame
    # Returns: tag images, lifetime data, and pixel-wise TCSPC histograms
    tag, tau, tcspc_pix = Process_Frame(
        im_sync[ind], im_col[ind], im_line[ind], im_chan[ind],
        im_tcspc[ind], head, cnum=cnum, resolution=resolution
    )
    
    # ========================================================================
    # MEMBRANE CHANNEL IMAGE RECONSTRUCTION
    # ========================================================================
    
    # Find optimal time gate position for membrane channel
    # This identifies the peak fluorescence timing for membrane staining
    pos = np.argmax(np.sum(tcspc_pix[:, :, :, mem_det*mem_PIE-1], axis=(0, 1)))
    nCh = pos + int(np.ceil(5/resolution))  # Gate width: ~5ns
    
    # Create time-gated membrane image by summing photons in optimal time window
    img_mem = np.sum(tcspc_pix[:, :, pos:pos + nCh-1, mem_det*mem_PIE-1], axis=2)
    
    # Alternative reconstruction methods (currently commented):
    # - Direct TAG image: img_mem = tag[:,:,mem_det-1,mem_PIE-1]
    # - Filtered by specific decay values
    # - Other time-gating strategies
    
    print(f"Membrane image reconstructed: {img_mem.shape}, intensity range: {img_mem.min()}-{img_mem.max()}")
    
    # ========================================================================
    # IMAGE PREPROCESSING FOR SEGMENTATION
    # ========================================================================
    
    # Normalize image for neural network input (0-1 range)
    img_np_scaled = (img_mem - np.min(img_mem)).astype(float)
    img_np_scaled /= np.max(img_np_scaled)
    
    # ========================================================================
    # DEEP LEARNING-BASED CELL SEGMENTATION
    # ========================================================================
    
    # Apply U-Net model for initial cell boundary prediction
    # Model: "confocal_2D_unet_ovules_ds2x" - specialized for plant cell segmentation
    pred = unet_predictions(
        img_np_scaled[np.newaxis, :, :],  # Add batch dimension
        "confocal_2D_unet_ovules_ds2x",   # Pre-trained model name
        'pioneering-rhino',               # Model repository/version
        patch=[1, 512, 512]               # Patch size for processing
    )
    
    print(f"U-Net prediction completed: {pred.shape}")
    
    # ========================================================================
    # PARAMETER OPTIMIZATION FOR SEGMENTATION QUALITY
    # ========================================================================
    
    # Test different parameter combinations to find optimal segmentation
    res = []  # Store results for each parameter combination
    
    for param in tqdm(params, desc="Optimizing segmentation parameters"):
        beta = param["beta"]
        post_mini_size = param["post_minsize"]
        
        # Apply mutex watershed segmentation with current parameters
        # mutex_ws: Multi-class watershed with GASP (Graph-based Active Segmentation)
        # Beta parameter controls under/over-segmentation trade-off
        mask = mutex_ws(
            pred,                    # Input prediction from U-Net
            superpixels=None,        # No pre-computed superpixels
            beta=beta,               # GASP parameter (0.5=under-seg, 0.95=over-seg)
            post_minsize=post_mini_size,  # Minimum segment size
            n_threads=6              # Parallel processing threads
        )
        
        # Alternative segmentation method (currently commented):
        # mask = dt_watershed(pred[0,:,:], n_threads=6)
        
        # Store segmentation result with parameters
        res.append({
            "name": f'frame_{nz}_beta_{beta}_minsize_{post_mini_size}',
            "beta": beta,
            "post_mini_size": post_mini_size,
            "pred": pred[0, :, :],   # U-Net prediction
            "mask": mask,            # Final segmentation mask
            # Future extensions:
            # "overlay": imgout,     # Overlay visualization
            # "props": props,        # Region properties
            # "props_df": props_df   # Properties dataframe
        })
        
    tmp = [np.unique(d['mask'], return_counts=True) for d in res]
    ncells = [len(unique) for unique, counts in tmp] # number of cells detected for each parameter
    idx = np.argmax(ncells) # gives the first position of maximum cells detected
    beta = res[idx]['beta']
    post_mini_size = res[idx]['post_mini_size']
    CellId, CellArea = tmp[idx]
    cidx = CellArea<max_cell_Area
    CellId = CellId[cidx]
    CellArea = CellArea[cidx]
    Cnum = np.argsort(CellId)
    maskt = res[idx]["mask"][0]
    mask = 0*maskt
    ncells = len(Cnum) # number of cells less than the max_cell_Area
    for j in range(ncells):
        mask[maskt == CellId[Cnum[j]]] = j+1
    
    im_mask = mask[im_line, im_col] # this is now a vector that assigns a mask value to each photon dependin on its im_col and im_line
    tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution']) # total number of channels in the original tcspc histogram
    
    ind = (im_chan == dind[auto_det-1]) & (im_tcspc<tmpCh/cnum*auto_PIE) & (im_tcspc>=((auto_PIE-1))*tmpCh/cnum)
    idx = np.where(ind)[0]
    
    
    Resolution = max(head['MeasDesc_Resolution'] * 1e9, resolution)  # resolution of 0.256 ns to calculate average lifetimes
    chDiv = np.ceil(1e-9 * Resolution / head['MeasDesc_Resolution'])
    SyncRate = 1.0 / head['MeasDesc_GlobalResolution']
    Ngate = round(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'] * (head['MeasDesc_Resolution'] / Resolution / cnum) * 1e9)
    # tcspc_cell =  np.zeros((len(Cnum), Ngate), dtype=np.uint32)
    # print(len(idx))
    if len(idx) > 0:  # Check if there are photons to process
        tcspc_cell = mHist2(im_mask[idx].astype(np.int64),
                                    (im_tcspc[idx] / chDiv).astype(np.int64) - int((auto_PIE-1)*tmpCh/cnum/chDiv),
                                    np.arange(len(Cnum)),
                                    np.arange(Ngate))[0]  # tcspc histograms for all the pixels at once!
    else:
        print(f"Warning: No photons found for frame {nz} - skipping FLIM analysis")
        continue  # Skip to next frame

    tcspcIRF = Calc_mIRF(head, np.sum(tcspc_cell,axis=0)[np.newaxis,:,np.newaxis]);
    tmpi = np.where((tcspcIRF/np.max(tcspcIRF))<(10**-4))[1]
    tcspcIRF[:,tmpi,:]=0
    
    
    # we are assuming 3 exponents for each cell in what follows:
    tauCell = np.zeros((ncells,3)) 
    ACell   = np.zeros_like(tauCell)
    LIm  = np.zeros((*img_mem.shape,3)) # lifetime image with three planes
    AIm  = np.zeros((*img_mem.shape,3)) # amplitude image with three planes
    for c in range(ncells):
        tau0 = np.array([0.5, 2.0, 5.0]) # initial guesses        
        taufit, A, _, _, _, _, _, _, _ = FluoFit(np.squeeze(tcspcIRF), \
                                                            np.squeeze(tcspc_cell[c,:]), \
                                                            np.floor(head['MeasDesc_GlobalResolution']*10**9/cnum + 0.5), \
                                                            resolution, tau0 )
        tauCell[c,:] = taufit
        ACell[c,:] = A
        sidx = np.argsort(tauCell[c,:])
        tauCell[c,:] = tauCell[c,sidx] # sorted lifetimes
        ACell[c,:] = ACell[c,sidx]/np.sum(ACell[c,:]) # normalized amplitudes
        indimm = mask==c+1 # pixels where the cell is present
        LIm[indimm,:] = tauCell[c,:]
        AIm[indimm,:] = ACell[c,:]
    #     taufit, A, cc, z, zz, offset, irs, t, chi = FluoFit(np.squeeze(tcspcIRF), \
    #                                                           np.squeeze(tcspc_cell[c,:]), \
    #                                                           np.floor(head['MeasDesc_GlobalResolution']*10**9/cnum + 0.5), \
    #                                                           resolution, tau0)
    fig, axs = plt.subplots(1,3,figsize=(20,10))
    axs = axs.ravel()
    fig.suptitle(res[idx]["name"])
    
    axs[0].imshow(img_mem,cmap="gray")
    axs[0].set_title('Rescaled Input')
    axs[0].axis('off')
    
    axs[1].imshow(res[idx]["pred"])
    axs[1].set_title('PlantSeg Pred')
    axs[1].axis('off')
    
    axs[2].imshow(res[idx]["mask"][0])
    axs[2].set_title('Label')
    axs[2].axis('off')
    
 # Here comes the plotting of the lifetime values and the amplitudes
            
 
# ============================================================================
# ADDITIONAL ANALYSIS SECTIONS (EXPERIMENTAL/TESTING)
# ============================================================================

# The following sections contain experimental analysis code and testing routines
# These are primarily used for method development and validation

# ============================================================================
# ALTERNATIVE FITTING METHOD: DISTRIBUTED FLUORESCENCE LIFETIME FITTING
# ============================================================================

# Test distributed lifetime fitting on first cell (experimental approach)
# This method can provide lifetime distributions rather than discrete components
if 'tcspc_cell' in locals() and 'tcspcIRF' in locals() and 'ncells' in locals():
    print("\n--- Testing Distributed Lifetime Fitting ---")
    
    if ncells > 0 and tcspc_cell.shape[0] > 0:
        try:
            cx, tau_dist, offset, c_dist, _, _, _ = DistFluoFit(
                np.squeeze(tcspc_cell[0, :]),                              # First cell's histogram
                np.floor(head['MeasDesc_GlobalResolution']*1e9/cnum + 0.5), # Repetition period
                resolution,                                                # Time resolution
                np.squeeze(tcspcIRF)                                      # Instrument response
            )
            
            print(f"Distributed fitting results:")
            print(f"Lifetime distribution centers: {tau_dist}")
            print(f"Distribution coefficients: {cx}")
            print(f"Background offset: {offset}")
            
        except Exception as e:
            print(f"Distributed fitting failed: {e}")
    else:
        print("No cells available for distributed lifetime fitting")

# ============================================================================
# IRF VALIDATION AND QUALITY CHECK
# ============================================================================

if 'tcspcIRF' in locals():
    print("\n--- IRF Quality Assessment ---")
    
    # Re-apply noise filtering to IRF for validation
    tmpi = np.where((tcspcIRF/np.max(tcspcIRF)) < (10**-4))[1]
    tcspcIRF[:, tmpi, :] = 0
    
    print(f"IRF shape: {tcspcIRF.shape}")
    print(f"IRF max value: {np.max(tcspcIRF):.0f} counts")
    print(f"IRF FWHM estimate: ~{len(tmpi)} time bins filtered")

# ============================================================================
# SINGLE CELL DETAILED ANALYSIS (VALIDATION)
# ============================================================================

if 'ncells' in locals() and 'tcspc_cell' in locals() and 'tcspcIRF' in locals():
    print("\n--- Detailed Single Cell Analysis (Cell #3) ---")
    
    # Perform detailed fitting on cell index 2 (3rd cell) for validation
    if ncells > 2 and tcspc_cell.shape[0] > 2:
        cell_idx = 2  # Third cell (0-indexed)
        tau0 = np.array([0.5, 2.0, 5.0])  # Initial lifetime guesses
        
        try:
            taufit_test, A_test, _, _, _, _, _, _, _ = FluoFit(
                np.squeeze(tcspcIRF),                                   # IRF
                np.squeeze(tcspc_cell[cell_idx, :]),                    # Cell histogram
                np.floor(head['MeasDesc_GlobalResolution']*1e9/cnum + 0.5), # Period
                resolution,                                             # Resolution
                tau0                                                   # Initial guesses
            )
            
            print(f"Test cell {cell_idx+1} fitting results:")
            print(f"Lifetimes: {taufit_test} ns")
            print(f"Amplitudes: {A_test}")
            print(f"Normalized amplitudes: {A_test/np.sum(A_test)}")
            
            # Compare with main analysis results
            if 'tauCell' in locals() and 'ACell' in locals() and tauCell.shape[0] > cell_idx:
                print(f"Main analysis lifetimes: {tauCell[cell_idx, :]} ns")
                print(f"Main analysis amplitudes: {ACell[cell_idx, :]}")
            
        except Exception as e:
            print(f"Test fitting failed: {e}")
    else:
        print("Insufficient cells for detailed validation (need >2 cells)")

print("\n" + "="*80)
print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)
print(f"Total frames processed: {nFrames}")
if 'ncells' in locals():
    print(f"Final cell count: {ncells}")
if 'tauCell' in locals():
    print(f"Lifetime components fitted: {tauCell.shape[1]}")
    print(f"Results saved in variables: tauCell, ACell, LIm, AIm")
else:
    print("Lifetime analysis not completed - check for errors in processing")
print("="*80)

# ============================================================================
# UTILITY FUNCTIONS FOR RESULTS ANALYSIS
# ============================================================================

def summarize_flim_results(tauCell: np.ndarray, ACell: np.ndarray,
                          ncells: int) -> None:
    """
    Generate a comprehensive summary of FLIM analysis results.
    
    Parameters:
    -----------
    tauCell : np.ndarray
        Lifetime values for each cell [ncells x n_components]
    ACell : np.ndarray
        Amplitude coefficients for each cell [ncells x n_components]
    ncells : int
        Number of analyzed cells
        
    Returns:
    --------
    None (prints summary to console)
    """
    print("\n" + "="*60)
    print("FLIM ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Number of cells analyzed: {ncells}")
    print(f"Lifetime components per cell: {tauCell.shape[1]}")
    
    # Overall lifetime statistics
    print(f"\nLifetime Statistics (ns):")
    print(f"  Minimum lifetime: {np.min(tauCell):.3f}")
    print(f"  Maximum lifetime: {np.max(tauCell):.3f}")
    print(f"  Mean lifetime: {np.mean(tauCell):.3f}")
    print(f"  Std deviation: {np.std(tauCell):.3f}")
    
    # Component-wise statistics
    print(f"\nComponent-wise Analysis:")
    for i in range(tauCell.shape[1]):
        print(f"  Component {i+1}:")
        print(f"    Lifetime range: {np.min(tauCell[:,i]):.3f} - {np.max(tauCell[:,i]):.3f} ns")
        print(f"    Mean amplitude: {np.mean(ACell[:,i]):.3f} ± {np.std(ACell[:,i]):.3f}")
    
    # Cell-by-cell summary (first 5 cells)
    print(f"\nIndividual Cell Summary (first 5 cells):")
    for c in range(min(5, ncells)):
        print(f"  Cell {c+1}: τ=[{', '.join([f'{t:.2f}' for t in tauCell[c,:]])}] ns, "
              f"A=[{', '.join([f'{a:.3f}' for a in ACell[c,:]])}]")
    
    if ncells > 5:
        print(f"  ... and {ncells-5} more cells")
    
    print("="*60)

def save_flim_results(filename: str, tauCell: np.ndarray, ACell: np.ndarray,
                     LIm: np.ndarray, AIm: np.ndarray, mask: np.ndarray,
                     analysis_params: Dict[str, Any]) -> None:
    """
    Save FLIM analysis results to a pickle file.
    
    Parameters:
    -----------
    filename : str
        Output filename for saving results
    tauCell, ACell : np.ndarray
        Cell-wise lifetime and amplitude data
    LIm, AIm : np.ndarray
        Pixel-wise lifetime and amplitude images
    mask : np.ndarray
        Final segmentation mask
    analysis_params : dict
        Dictionary containing analysis parameters
    """
    results = {
        'tauCell': tauCell,
        'ACell': ACell,
        'LIm': LIm,
        'AIm': AIm,
        'mask': mask,
        'ncells': len(tauCell),
        'analysis_params': analysis_params,
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {filename}")

# Generate summary if analysis was completed
if 'tauCell' in locals() and 'ACell' in locals() and 'ncells' in locals():
    summarize_flim_results(tauCell, ACell, ncells)
    
    # Optional: Save results
    # save_filename = filename.replace('.ptu', '_FLIM_results.pkl')
    # analysis_params = {
    #     'resolution': resolution,
    #     'max_cell_Area': max_cell_Area,
    #     'mem_PIE': mem_PIE, 'mem_det': mem_det,
    #     'auto_PIE': auto_PIE, 'auto_det': auto_det,
    #     'optimal_beta': beta,  # Use the actual beta variable from the loop
    #     'optimal_post_mini_size': post_mini_size  # Use the actual post_mini_size variable
    # }
    # save_flim_results(save_filename, tauCell, ACell, LIm, AIm, mask, analysis_params)
