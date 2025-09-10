# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:01:28 2025

FlavMetaFLIM.py - FLIM Analysis Pipeline for FAD metabolic imaging

@author: narain karedla
"""

# Standard library imports
import numpy as np
import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from tifffile import imwrite

# Third-party imports
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Custom module imports
from PTU_ScanRead import PTU_ScanRead, Process_Frame, mHist, cim
from FLIM_fitter import Calc_mIRF, FluoFit, PatternMatchIm

#%%
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def vignette_correction(int_im, blur_factor=6, lower_percentile=2, upper_percentile=98):
    """
    Apply vignette correction to intensity image following MATLAB algorithm.
    
    This function corrects for illumination non-uniformity by:
    1. Creating a blurred background estimate using Gaussian filter
    2. Normalizing the original image by this background
    3. Applying percentile-based contrast enhancement
    4. Final normalization to [0,1] range
    
    MATLAB equivalent:
    bblur = imgaussfilt(int_im, min(size(int_im))/6);
    bblur = bblur ./ max(bblur(:));
    I2 = int_im ./ bblur;
    lb = prctile(I2(:), 2);
    ub = prctile(I2(:), 98);
    I2(I2 < lb) = lb;
    I2(I2 > ub) = ub;
    I2 = I2 - lb;
    int2 = I2 ./ max(I2(:));
    
    Parameters
    ----------
    int_im : ndarray
        Input intensity image
    blur_factor : float, optional (default=6)
        Gaussian blur factor = min(image_size) / blur_factor
    lower_percentile : float, optional (default=2)
        Lower percentile for contrast adjustment (removes dark outliers)
    upper_percentile : float, optional (default=98)
        Upper percentile for contrast adjustment (removes bright outliers)
        
    Returns
    -------
    int2 : ndarray
        Vignette-corrected and normalized intensity image [0,1]
    bblur : ndarray
        Gaussian-blurred background estimate
    """
    
    print("Applying vignette correction...")
    print(f"  - Image shape: {int_im.shape}")
    print(f"  - Blur factor: 1/{blur_factor} of min image dimension")
    
    # Calculate Gaussian blur sigma: min(image_size) / blur_factor
    sigma = min(int_im.shape) / blur_factor
    print(f"  - Gaussian sigma: {sigma:.2f} pixels")
    
    # Create blurred background (excitation floor estimate)
    # MATLAB: bblur = imgaussfilt(int_im, min(size(int_im))/6);
    bblur = gaussian_filter(int_im, sigma=sigma)
    
    # Normalize blurred background to [0,1] with max=1
    # MATLAB: bblur = bblur ./ max(bblur(:));
    bblur = bblur / np.max(bblur)
    
    # Vignette correction: divide by normalized background
    # MATLAB: I2 = int_im ./ bblur;
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    I2 = int_im / (bblur + epsilon)
    
    # Calculate percentile bounds for contrast enhancement
    # MATLAB: lb = prctile(I2(:), 2); ub = prctile(I2(:), 98);
    lb = np.percentile(I2, lower_percentile)  # Lower bound (2nd percentile)
    ub = np.percentile(I2, upper_percentile)  # Upper bound (98th percentile)
    
    print(f"  - Intensity range before clipping: [{np.min(I2):.3f}, {np.max(I2):.3f}]")
    print(f"  - Percentile bounds: [{lb:.3f}, {ub:.3f}]")
    
    # Clip values to percentile bounds
    # MATLAB: I2(I2 < lb) = lb; I2(I2 > ub) = ub;
    I2 = np.clip(I2, lb, ub)
    
    # Subtract lower bound (baseline correction)
    # MATLAB: I2 = I2 - lb;
    I2 = I2 - lb
    
    # Final normalization to [0,1]
    # MATLAB: int2 = I2 ./ max(I2(:));
    int2 = I2 / np.max(I2) if np.max(I2) > 0 else I2
    
    print(f"  - Final intensity range: [{np.min(int2):.3f}, {np.max(int2):.3f}]")
    
    return int2, bblur

def subplot_cim(x, brightness=None, color_range=None, colormap=None, ax=None):
    """
    Create a cim-style display within a matplotlib subplot.
    
    This function replicates the cim functionality but works within subplots,
    creating an RGB image with color determined by x and brightness overlay.
    
    Parameters
    ----------
    x : ndarray
        Color data (2D array)
    brightness : ndarray, optional
        Brightness overlay data (same shape as x)
    color_range : tuple, optional
        (min, max) range for color scaling
    colormap : ndarray, optional
        Custom colormap (64×3 RGB values)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes
        
    Returns
    -------
    handle : matplotlib image handle
        Handle to the displayed image
    """
    
    if ax is None:
        ax = plt.gca()
    
    x = np.asarray(x, dtype=float)
    
    # Simple display if no brightness overlay
    if brightness is None:
        if color_range is not None:
            handle = ax.imshow(x, cmap='viridis', vmin=color_range[0], vmax=color_range[1])
        else:
            handle = ax.imshow(x, cmap='viridis')
        ax.axis('off')
        return handle
    
    # Create brightness overlay
    brightness = np.asarray(brightness, dtype=float)
    
    # Normalize brightness to [0, 1]
    valid_brightness = np.isfinite(brightness)
    if np.any(valid_brightness):
        b_min = np.min(brightness[valid_brightness])
        b_max = np.max(brightness[valid_brightness])
        if b_max > b_min:
            brightness = (brightness - b_min) / (b_max - b_min)
        else:
            brightness = np.zeros_like(brightness)
    brightness = np.clip(brightness, 0, 1)
    
    # Handle color scaling
    if color_range is not None:
        x_min, x_max = color_range
    else:
        valid_x = np.isfinite(x) & valid_brightness
        if np.any(valid_x):
            x_min = np.min(x[valid_x])
            x_max = np.max(x[valid_x])
        else:
            x_min, x_max = 0, 1
    
    # Scale x to colormap indices [0, 1]
    if x_max > x_min:
        x_scaled = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    else:
        x_scaled = np.full_like(x, 0.5)
    
    # Set up colormap
    if colormap is None:
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, 256))[:, :3]
    else:
        colors = np.asarray(colormap)
        if colors.shape[0] != 256:
            # Interpolate to 256 colors for better resolution
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, colors.shape[0])
            new_indices = np.linspace(0, 1, 256)
            colors_interp = []
            for i in range(3):
                f = interp1d(old_indices, colors[:, i], kind='linear')
                colors_interp.append(f(new_indices))
            colors = np.column_stack(colors_interp)
    
    # Create RGB image
    h, w = x.shape
    rgb_image = np.zeros((h, w, 3))
    
    # Handle NaN values
    x_scaled = np.nan_to_num(x_scaled, nan=0)
    brightness = np.nan_to_num(brightness, nan=0)
    
    # Map to colormap and apply brightness
    color_indices = (x_scaled * (colors.shape[0] - 1)).astype(int)
    color_indices = np.clip(color_indices, 0, colors.shape[0] - 1)
    
    for channel in range(3):
        rgb_image[:, :, channel] = colors[color_indices, channel] * brightness
    
    # Display the image
    handle = ax.imshow(rgb_image)
    ax.axis('off')
    
    # Add colorbar as scale bar
    if color_range is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Create a colorbar using the colormap
        if colormap is None:
            cmap = plt.cm.viridis
        else:
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(colormap)
        
        # Create a scalar mappable for the colorbar
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=x_min, vmax=x_max)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=8)
    
    return handle

def display_amplitude_maps_subplot(Amp, int2, taufit, flag_win=True, colormap=None, save_path=None):
    """
    Display all amplitude maps in a single figure with subplots using cim-style visualization.
    
    Parameters
    ----------
    Amp : ndarray
        Amplitude data (nx, ny, n_components)
    int2 : ndarray
        Vignette-corrected intensity for brightness overlay
    taufit : array-like
        Fitted lifetime values
    flag_win : bool, optional
        Flag indicating windowed vs pixel-wise analysis
    colormap : ndarray, optional
        Custom colormap (64×3 RGB values)
    save_path : str, optional
        Path to save the figure
    """
    
    print(f"Creating amplitude subplot display for {len(taufit)} lifetime components...")
    
    # Create custom colormap if not provided
    if colormap is None:
        import matplotlib.cm as cm
        colormap = cm.viridis(np.linspace(0, 1, 64))[:, :3]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Display background/offset amplitude (component 0)
    amp_range = [np.min(Amp[:, :, 0]), np.max(Amp[:, :, 0])]
    subplot_cim(Amp[:, :, 0], int2**0.5, amp_range, colormap, ax=axes[0])
    axes[0].set_title('Background/Offset', fontsize=12, pad=10)
    
    # Display amplitudes for each lifetime component
    for i, tau in enumerate(taufit):
        if i + 1 < len(axes):
            amp_data = Amp[:, :, i + 1]
            amp_range = [np.min(amp_data), np.max(amp_data)]
            subplot_cim(amp_data, int2**0.5, amp_range, colormap, ax=axes[i + 1])
            axes[i + 1].set_title(f'Component {i+1}: τ = {tau:.2f} ns', fontsize=12, pad=10)
    
    # Hide unused subplot if only 2 lifetimes
    if len(taufit) < 3:
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'FLIM Amplitude Maps - {"Windowed" if flag_win else "Pixel-wise"} Analysis',
                 fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Amplitude maps saved to: {save_path}")
    
    plt.show()
    
    print(f" All {len(taufit)+1} amplitude components displayed in single subplot figure")

#%%
# ============================================================================
# CORE FLIM ANALYSIS FUNCTIONS
# ============================================================================

def load_or_process_ptu_data(filename: str, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Load PTU data from cache or process raw PTU file.
    
    Parameters
    ----------
    filename : str
        Path to PTU file
    force_reprocess : bool, optional
        Force reprocessing even if cache exists
        
    Returns
    -------
    dict
        Dictionary containing FLIM data arrays and header
    """
    
    res_file = filename[:-4] + '_FLIM_data.pkl'
    
    if os.path.exists(res_file) and not force_reprocess:
        print(f"Loading cached FLIM data from: {res_file}")
        with open(res_file, 'rb') as f:
            FLIM_data = pickle.load(f)
        
        # Extract data
        data = {
            'im_sync': FLIM_data['im_sync'],
            'im_tcspc': FLIM_data['im_tcspc'],
            'im_chan': FLIM_data['im_chan'],
            'im_line': FLIM_data['im_line'],
            'im_frame': FLIM_data['im_frame'],
            'im_col': FLIM_data['im_col'],
            'head': FLIM_data['head']
        }
        del FLIM_data
    else:
        print(f"Processing raw PTU file: {filename}")
        head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead(filename)
        
        data = {
            'im_sync': im_sync,
            'im_tcspc': im_tcspc,
            'im_chan': im_chan,
            'im_line': im_line,
            'im_frame': im_frame,
            'im_col': im_col,
            'head': head
        }
        del head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame
    return data

def analyze_flim_data(data: Dict[str, Any], 
                     auto_det: int = 0, 
                     auto_PIE: int = 1, 
                     flag_win: bool = True,
                     resolution: float = 0.2,
                     tau0: np.ndarray = np.array([0.3, 1.7, 6.0]),
                     win_size: int = 8,
                     step: int = 2,
                     IRF_data = None) -> Dict[str, Any]:
    """
    Perform FLIM analysis on PTU data.
    
    Parameters
    ----------
    data : dict
        PTU data dictionary from load_or_process_ptu_data
    auto_det : int, optional
        Detector ID for autofluorescence channel detection
    auto_PIE : int, optional
        Laser pulse number for autofluorescence channel (PIE window)
    flag_win : bool, optional
        Flag for window based lifetime estimation
    resolution : float, optional
        Temporal resolution for TCSPC histogram binning (ns)
    tau0 : ndarray, optional
        Initial lifetime guesses
    win_size : int, optional
        Size of the sliding window (pixels)
    step : int, optional
        Step size for window movement
        
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    
    # Extract data
    # CRITICAL FIX: Use appropriate integer data types to save memory and ensure correct data handling
    im_sync = np.array(data['im_sync'], dtype=np.uint32)
    im_tcspc = np.array(data['im_tcspc'], dtype=np.uint16)
    im_chan = np.array(data['im_chan'], dtype=np.uint8)
    im_line = np.array(data['im_line'], dtype=np.uint16)
    im_col = np.array(data['im_col'], dtype=np.uint16)
    head = data['head']
    
    del data
    # PIE configuration
    cnum = 1
    if 'PIENumPIEWindows' in head:
        cnum = head['PIENumPIEWindows']
        print(f"PIE cycles detected: {cnum}")
    
    # Process frame
    print("Processing FLIM data...")
    tag, tau, tcspc_pix = Process_Frame(
        im_sync, im_col, im_line, im_chan,
        im_tcspc, head, cnum=cnum, resolution=resolution
    )
    
    nx, ny, ch, p = np.shape(tag)
    
    # Calculate parameters
    Resolution = max(head['MeasDesc_Resolution'] * 1e9, resolution)
    chDiv = np.ceil(1e-9 * Resolution / head['MeasDesc_Resolution'])
    #SyncRate = 1.0 / head['MeasDesc_GlobalResolution']
    Ngate = round(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'] * 
                  (head['MeasDesc_Resolution'] / Resolution / cnum) * 1e9)
    tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'])
    
    # Extract channel data
    idx = im_chan == auto_det
    if np.sum(idx) > 0:
        tcspc_im = mHist((im_tcspc[idx] / chDiv).astype(np.int64) - 
                        int((auto_PIE-1)*tmpCh/cnum/chDiv),
                        np.arange(Ngate))[0]
    else:
        print(f"Warning: No photons found for channel {auto_det} - skipping FLIM analysis")
        return {}
    
    # Calculate IRF and perform fitting
    if type(IRF_data) is dict:
        # Loading IRF provided as input
        tcspcIRF = np.array(IRF_data['tcspcIRF'], dtype=np.float64)
        # TODO: Check resolution of IRF file before accepting it
    else:
        print("Calculating IRF and performing lifetime fitting...")
        tcspcIRF = Calc_mIRF(head, tcspc_im[np.newaxis, :, np.newaxis])
        tmpi = np.where((tcspcIRF/np.max(tcspcIRF)) < (10**-4))[1]
        tcspcIRF[:, tmpi, :] = 0
        
    tmp_tau = tau0.copy()
    taufit, A, _, zfit, patterns, _, _, _, _ = FluoFit(
        np.squeeze(tcspcIRF), tcspc_im,
        np.floor(head['MeasDesc_GlobalResolution']*10**9/cnum + 0.5),
        resolution, tmp_tau, flag_ml=False
    )
    
    # Sort lifetime values and reorder corresponding patterns and amplitudes
    sort_idx = np.argsort(taufit)
    taufit = taufit[sort_idx]
    A = A[sort_idx]
    patterns = patterns[:, 1:]  # Remove background pattern (first column)
    patterns = patterns[:, sort_idx]  # Reorder patterns
    patterns = np.column_stack([np.ones(patterns.shape[0]), patterns])  # Add background back as first column
    patterns = patterns / np.sum(patterns, axis=0)  # normalized patterns
    
    print(f"Sorted lifetime values: {taufit}")
    
    # Pattern matching analysis
    if flag_win:
        print(f"Processing with sliding window: {win_size}x{win_size}, step={step}")
        
        # Calculate output dimensions
        n_win_x = (nx - win_size) // step + 1
        n_win_y = (ny - win_size) // step + 1
        
        print(f"Original image: {nx}x{ny}, Windows: {n_win_x}x{n_win_y}")
        
        # Initialize arrays
        tcspc_win = np.zeros((n_win_x, n_win_y, tcspc_pix.shape[2]))
        int_im = np.zeros((n_win_x, n_win_y))
        
        print("Aggregating TCSPC data with sliding windows...")
        for i in range(n_win_x):
            for j in range(n_win_y):
                x_start = i * step
                x_end = x_start + win_size
                y_start = j * step
                y_end = y_start + win_size
                
                tcspc_win[i, j, :] = np.sum(tcspc_pix[x_start:x_end, y_start:y_end, :, 0], axis=(0, 1))
                int_im[i, j] = np.sum(tcspc_win[i, j, :])
        
        print("Running pattern matching on windowed data...")
        Amp, Z = PatternMatchIm(tcspc_win, patterns, mode='PIRLS')
        print(f"Windowed analysis completed: {n_win_x}x{n_win_y} windows processed")
        
    else:
        # Pixel by pixel analysis
        print("Running pixel-by-pixel pattern matching...")
        Amp, Z = PatternMatchIm(tcspc_pix[:, :, :, 0], patterns, mode='PIRLS')
        int_im = np.sum(tcspc_pix[:, :, :, 0], axis=2)
    
    # Normalize amplitudes
    Amp = Amp / np.sum(Amp[:, :, 1:], axis=2, keepdims=True)
    
    # Apply vignette correction
    print("Applying vignette correction...")
    int2, bblur = vignette_correction(int_im)
    
    # Create average lifetime image
    tau_avg_int = np.zeros_like(int_im)
    for i, tau_val in enumerate(taufit):
        tau_avg_int += Amp[:, :, i + 1] * tau_val
    
    #print(np.sum(Amp[:, :, 1:],axis=2))
    #intensity_safe = np.where(int2 > 0, int2, 1)
    tau_avg_int = np.where(int2 > 0, tau_avg_int / np.sum(Amp[:, :, 1:],axis=2), 0) # this is intensity averaged
    
    
    # 2. Rate-averaged lifetime
    rate_const = 1.0/taufit
    tau_avg_rate = np.zeros_like(int_im)
    for i, rate in enumerate(rate_const):
        tau_avg_rate += Amp[:, :, i + 1] * rate
    
    # Handle division by zero and invalid values in rate averaging
    tau_avg_rate = np.where(tau_avg_rate > 0, np.sum(Amp[:, :, 1:],axis=2)/tau_avg_rate, 0)
    
    # Print statistics
    print("\nLifetime Statistics:")
    valid_int = tau_avg_int > 0
    valid_rate = tau_avg_rate > 0
    print(f"Intensity-averaged lifetime: min={np.min(tau_avg_int[valid_int]):.2f}ns, max={np.max(tau_avg_int[valid_int]):.2f}ns, mean={np.mean(tau_avg_int[valid_int]):.2f}ns")
    print(f"Rate-averaged lifetime: min={np.min(tau_avg_rate[valid_rate]):.2f}ns, max={np.max(tau_avg_rate[valid_rate]):.2f}ns, mean={np.mean(tau_avg_rate[valid_rate]):.2f}ns")
    
    print("\nAmplitude Statistics:")
    print(f"Background: min={np.min(Amp[:,:,0]):.3f}, max={np.max(Amp[:,:,0]):.3f}, mean={np.mean(Amp[:,:,0]):.3f}")
    for i, tau in enumerate(taufit):
        amp_comp = Amp[:, :, i + 1]
        print(f"Component {i+1} (τ={tau:.2f}ns): min={np.min(amp_comp):.3f}, max={np.max(amp_comp):.3f}, mean={np.mean(amp_comp):.3f}")
    
    return {
        'tcspcIRF': tcspcIRF,
        'taufit': taufit,
        'patterns': patterns,
        'Amp': Amp,
        'int2': int2,
        'int_im': int_im,
        'tau_avg_int': tau_avg_int,
        'tau_avg_rate': tau_avg_rate,
        'flag_win': flag_win,
        'head': head
    }

def save_flim_results(results: Dict[str, Any], output_dir: str, filename_base: str):
    """
    Save FLIM analysis results in multiple formats.
    
    Parameters
    ----------
    results : dict
        Results dictionary from analyze_flim_data
    output_dir : str
        Output directory path
    filename_base : str
        Base filename for output files
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    
    # Save as pickle
    pkl_file = output_path / f"{filename_base}_FLIM_results.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f" Pickle file saved: {pkl_file}")
    
    # Save lifetime values as CSV
    csv_file = output_path / f"{filename_base}_lifetimes.csv"
    lifetime_data = {
        'Component': [f'τ{i+1}' for i in range(len(results['taufit']))],
        'Lifetime_ns': results['taufit']
    }
    df_lifetimes = pd.DataFrame(lifetime_data)
    df_lifetimes.to_csv(csv_file, index=False)
    print(f" Lifetime CSV saved: {csv_file}")
    
    # Save amplitude statistics as CSV
    stats_file = output_path / f"{filename_base}_amplitude_stats.csv"
    Amp = results['Amp']
    stats_data = []
    
    # Background statistics
    stats_data.append({
        'Component': 'Background',
        'Min': np.min(Amp[:, :, 0]),
        'Max': np.max(Amp[:, :, 0]),
        'Mean': np.mean(Amp[:, :, 0]),
        'Std': np.std(Amp[:, :, 0])
    })
    
    # Lifetime component statistics
    for i, tau in enumerate(results['taufit']):
        amp_comp = Amp[:, :, i + 1]
        stats_data.append({
            'Component': f'τ{i+1}_{tau:.2f}ns',
            'Min': np.min(amp_comp),
            'Max': np.max(amp_comp),
            'Mean': np.mean(amp_comp),
            'Std': np.std(amp_comp)
        })
    
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv(stats_file, index=False)
    print(f" Amplitude statistics CSV saved: {stats_file}")
    
    # Save images as TIFF
    # Intensity image
    int_tiff = output_path / f"{filename_base}_intensity.tif"
    imwrite(int_tiff, results['int2'].astype(np.float32))
    print(f"Intensity TIFF saved: {int_tiff}")
    
    # Average lifetime images
    tau_int_tiff = output_path / f"{filename_base}_intensity_averaged_lifetime.tif"
    imwrite(tau_int_tiff, results['tau_avg_int'].astype(np.float32))
    print(f"Intensity-averaged lifetime TIFF saved: {tau_int_tiff}")
    
    tau_rate_tiff = output_path / f"{filename_base}_rate_averaged_lifetime.tif"
    imwrite(tau_rate_tiff, results['tau_avg_rate'].astype(np.float32))
    print(f"Rate-averaged lifetime TIFF saved: {tau_rate_tiff}")
    
    # Individual amplitude component TIFFs
    for i in range(results['Amp'].shape[2]):
        if i == 0:
            comp_name = "background"
        else:
            comp_name = f"component_{i}_tau_{results['taufit'][i-1]:.2f}ns"
        
        amp_tiff = output_path / f"{filename_base}_amplitude_{comp_name}.tif"
        imwrite(amp_tiff, results['Amp'][:, :, i].astype(np.float32))
        print(f"Amplitude TIFF saved: {amp_tiff}")
    
    # Display and save CIM lifetime images
    print("\nDisplaying FLIM images using subplot_cim function with vignette-corrected intensity...")
    
    # Create custom colormap (similar to MATLAB's jet)
    custom_jet_cmap = plt.cm.jet(np.linspace(0, 1, 64))[:, :3]
    
    # Create figure with subplots for both lifetime images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Intensity-averaged lifetime
    subplot_cim(results['tau_avg_int'], results['int2']**0.75, [1.5, 4], custom_jet_cmap, ax=ax1)
    ax1.set_title('Intensity-Averaged Lifetime', fontsize=12)
    
    # Rate-averaged lifetime
    subplot_cim(results['tau_avg_rate'], results['int2']**0.75, [1.5, 4], custom_jet_cmap, ax=ax2)
    ax2.set_title('Rate-Averaged Lifetime', fontsize=12)
    
    plt.suptitle('FLIM Images: Lifetime with Vignette-Corrected Intensity Overlay', fontsize=14)
    plt.tight_layout()
    
    # Save FLIM image
    flim_file = output_path / f"{filename_base}_FLIM_lifetime_comparison.png"
    plt.savefig(flim_file, dpi=300, bbox_inches='tight')
    print(f"FLIM lifetime comparison image saved: {flim_file}")
    plt.show()
    
    # Display and save amplitude visualization with cim-style subplots
    print("\nDisplaying amplitude maps...")
    display_amplitude_maps_subplot(
        results['Amp'], results['int2'], results['taufit'],
        results['flag_win'], custom_jet_cmap, save_path=None
    )
    
    # The amplitude visualization is already displayed above with the lifetime images

def process_single_file(ptu_file: str, **analysis_params) -> Dict[str, Any]:
    """
    Process a single PTU file with FLIM analysis.
    Results are saved in the same directory as the PTU file.
    
    Parameters
    ----------
    ptu_file : str
        Path to PTU file
    **analysis_params
        Additional parameters for analyze_flim_data
        
    Returns
    -------
    dict
        Analysis results
    """
    
    print(f"\n{'='*60}")
    print(f"Processing: {ptu_file}")
    print(f"{'='*60}")
    
    try:
        # Load/process data
        data = load_or_process_ptu_data(ptu_file)
        
        # Analyze
        results = analyze_flim_data(data, **analysis_params)
        
        if not results:
            print(f" Analysis failed for {ptu_file}")
            return {}
        
        # Save results in the same directory as the PTU file
        ptu_path = Path(ptu_file)
        output_dir = ptu_path.parent  # Same directory as PTU file
        filename_base = ptu_path.stem
        save_flim_results(results, str(output_dir), filename_base)
        
        print(f" Successfully processed: {ptu_file}")
        return results
        
    except Exception as e:
        print(f" Error processing {ptu_file}: {str(e)}")
        return {}

def process_multiple_folders(folder_paths: List[str], **analysis_params) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple folders containing PTU files.
    Results are saved in the same directories as the PTU files.
    
    Parameters
    ----------
    folder_paths : list of str
        List of folder paths containing PTU files
    **analysis_params
        Additional parameters for analyze_flim_data
        
    Returns
    -------
    dict
        Dictionary with folder names as keys and analysis results as values
    """
    
    all_results = {}
    
    for folder_path in folder_paths:
        folder_path = Path(folder_path)
        folder_name = folder_path.name
        
        print(f"\n{'='*80}")
        print(f"Processing folder: {folder_path}")
        print(f"{'='*80}")
        
        # Find all PTU files in folder
        ptu_files = list(folder_path.glob("*.ptu"))
        
        if not ptu_files:
            print(f" No PTU files found in {folder_path}")
            continue
        
        print(f"Found {len(ptu_files)} PTU files")
        
        folder_results = {}
        
        for ptu_file in ptu_files:
            # Process file and save results next to the PTU file
            file_results = process_single_file(str(ptu_file), **analysis_params)
            if file_results:
                folder_results[ptu_file.stem] = file_results
        
        all_results[folder_name] = folder_results
        
        print(f"\n Completed folder: {folder_name} ({len(folder_results)}/{len(ptu_files)} files processed)")
    
    return all_results

#%%
# ============================================================================
# EXAMPLE USAGE AND MAIN EXECUTION
# ============================================================================

IRF_file = r'D:\Collabs\fromKaitlyn\Kaitlyn\data\IRFmodel.pkl'
with open(IRF_file, 'rb') as f:
     IRF_data = pickle.load(f)
     
if __name__ == "__main__":
    # Example usage for single file
    single_file = False
    
    if single_file:
        # Single file processing - results saved next to PTU file
        ptu_file = r'D:\Collabs\fromKaitlyn\Kaitlyn\data\OTB5\RawImage1.ptu'
        
        results = process_single_file(
            ptu_file,
            auto_det=0,
            auto_PIE=1,
            flag_win=True,
            resolution=0.2,
            tau0=np.array([0.3, 1.7, 6.0]),
            win_size=8,
            step=2, 
            IRF_data = IRF_data
        )
    
    # Example usage for multiple folders
    else:
        # Multiple folder processing - results saved next to PTU files
        folder_paths = [
            r'D:\Collabs\fromKaitlyn\Kaitlyn\data\OTB5',
            # Add more folder paths as needed
        ]
        
        all_results = process_multiple_folders(
            folder_paths,
            auto_det=0,
            auto_PIE=1,
            flag_win=True,
            resolution=0.2,
            tau0=np.array([0.3, 1.7, 6.0]),
            win_size=8,
            step=2,
            IRF_data = IRF_data
        )
        
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total folders processed: {len(all_results)}")
        for folder_name, folder_results in all_results.items():
            print(f"  {folder_name}: {len(folder_results)} files")