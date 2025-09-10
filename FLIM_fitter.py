# -*- coding: utf-8 -*-
"""
FLIM (Fluorescence Lifetime Imaging Microscopy) Fitting Module

This module provides comprehensive tools for fitting fluorescence lifetime data,
including instrumental response function (IRF) modeling, convolution operations,
and various fitting algorithms for TCSPC (Time-Correlated Single Photon Counting) data.

Key Features:
- Convolution of IRF with decay functions
- IRF modeling with Gaussian and exponential components
- Multi-exponential decay fitting
- Maximum likelihood and least squares fitting
- Distributed lifetime fitting
- Non-negative least squares optimization

Created on Mon Sep  2 18:02:42 2024
@author: narai

Dependencies:
- numpy: Numerical operations
- matplotlib: Plotting
- scipy: Optimization and linear algebra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls, lsq_linear
from scipy.optimize import minimize as minimize_s
#from scipy.linalg import lstsq
# import cupy as cp
# from numba import cuda, float32
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
# from lmfit import minimize, Parameters #,report_fit, fit_report, report_errors,

# Try to import CUDA PIRLS functionality
try:
    from PIRLS_cu.pirls_pycuda import pirls_batch_cuda, CUDA_AVAILABLE
    if CUDA_AVAILABLE:
        print("✅ CUDA PIRLS module imported successfully - GPU acceleration available")
    else:
        print("⚠️  CUDA available but kernel compilation failed - using CPU-only implementation")
except ImportError:
    CUDA_AVAILABLE = False
    pirls_batch_cuda = None
    print("⚠️  CUDA PIRLS module not available - using CPU-only implementation")

#%%


def Convol(irf, x):
    """
    Convolves the instrumental response function (IRF) with decay function(s) using FFT.
    
    This function performs convolution in the frequency domain for computational efficiency.
    It handles periodic boundary conditions and can process multiple decay functions
    simultaneously.
    
    Parameters
    ----------
    irf : array-like
        Instrumental response function (1D array). This represents the system's
        response to an instantaneous excitation pulse.
    x : array-like
        Decay function(s). Can be 1D (single decay) or 2D (multiple decays).
        For 2D arrays, each column represents a different decay function.
    
    Returns
    -------
    y : ndarray
        Convolved result with the same shape as x. The convolution assumes
        periodic boundary conditions with period equal to the length of x.
    
    Notes
    -----
    - Background is estimated from the last 10 points of the IRF
    - IRF is padded or truncated to match the length of x
    - Convolution is performed using FFT for efficiency
    - Result is real-valued (imaginary parts from numerical errors are discarded)
    
    Examples
    --------
    >>> irf = np.exp(-np.arange(100)**2/100)  # Gaussian IRF
    >>> decay = np.exp(-np.arange(100)/10)    # Exponential decay
    >>> convolved = Convol(irf, decay)
    """
    # Convert inputs to numpy arrays and ensure proper shapes
    irf = np.array(irf).flatten()
    x = np.array(x)
    
    # Estimate background from the tail of the IRF (last 10 points)
    mm = np.mean(irf[-10:])
    
    # Ensure x is at least 2D for consistent processing
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    p = x.shape[0]  # Length of decay function(s)
    n = len(irf)    # Length of IRF
    
    # Adjust IRF length to match decay function length
    if p > n:
        # Pad IRF with background level if it's shorter than decay
        irf = np.concatenate([irf, mm * np.ones(p - n)])
    else:
        # Truncate IRF if it's longer than decay
        irf = irf[:p]
    
    # Perform convolution in frequency domain using FFT
    # FFT convolution: conv(a,b) = ifft(fft(a) * fft(b))
    y = np.real(np.fft.ifft(np.fft.fft(irf)[:, None] * np.fft.fft(x, axis=0), axis=0))
    
    # Handle periodic boundary conditions by selecting appropriate indices
    t = (np.arange(n) % p)
    y = y[t, :]
    
    # Remove extra dimensions if input was 1D
    return y.squeeze()

def IRF_Fun(p, t, pic=None):
    """
    Computes a model Instrumental Response Function (IRF) with multiple components.
    
    This function models the IRF as a combination of:
    1. Gaussian component (main peak)
    2. Exponential tail component (scattered light)
    3. Reconvolution component (detector response)
    
    The mathematical model is:
    IRF(t) = [Gaussian + b*Tail + a*Reconvolution] / Normalization
    
    Where:
    - Gaussian: exp(-(t-t₀)²/(2w²))
    - Tail: H(t-t₀-dt) * exp(-(t-t₀-dt)/T₁) * (1-exp(-(t-t₀-dt)/T₂))
    - H(x) is the Heaviside step function
    
    Parameters
    ----------
    p : array-like, shape (7,)
        Model parameters:
        p[0] = t_0  : Peak position (time offset)
        p[1] = w1   : Gaussian width parameter
        p[2] = T1   : First time constant for exponential component
        p[3] = T2   : Second time constant for exponential component
        p[4] = a    : Amplitude of reconvolution component
        p[5] = b    : Amplitude of tail component
        p[6] = dt   : Time delay for exponential component
    
    t : array-like
        Time vector at which to evaluate the IRF
        
    pic : int, optional
        Plotting option:
        - 1: Linear scale plot
        - Other values: Log scale plot
        - None: No plotting
    
    Returns
    -------
    z : ndarray
        Normalized IRF values at time points t
    
    References
    ----------
    Walther, K.A. et al, Mol. BioSyst. (2011) doi:10.1039/c0mb00132e
    
    Notes
    -----
    The IRF is automatically normalized to unit area. The function handles
    edge cases by clipping extreme values and ensuring monotonic rise before
    the peak.
    """
    # Ensure inputs are proper numpy arrays
    t = np.array(t).reshape(-1)
    p = np.array(p).reshape(-1)

    # Extract parameters with descriptive names
    t_0 = p[0]  # Peak position
    w1  = p[1]  # Gaussian width
    T1  = p[2]  # First time constant
    T2  = p[3]  # Second time constant
    a   = p[4]  # Reconvolution amplitude
    b   = p[5]  # Tail amplitude
    dt  = p[6]  # Time delay

    # Define shifted time variables
    t1 = t - t_0        # Time relative to peak
    t2 = t - t_0 - dt   # Time relative to delayed component
    t3 = t + t_0        # Time for pre-peak component

    # Heaviside step function for delayed component (only after t₀ + dt)
    H = np.ones(t.shape)
    H[t < (t_0 + dt)] = 0

    # Gate function for pre-peak component (only before t₀ - dt)
    G = np.zeros(t.shape)
    G[t < (t_0 - dt)] = 1

    # Compute the three IRF components
    IRF = np.array([
        np.exp(-t1**2 / (2 * w1)),                                    # Gaussian main peak
        G * np.exp(t3 / T1),                                          # Pre-peak tail
        H * (np.exp(-t2 / T1) * (1 - np.exp(-t2 / T2)))             # Post-peak reconvolution
    ])

    # Weight components and normalize
    # Apply amplitudes [1, b, a] to respective components
    weights = np.array([1, b, a])
    IRF_weighted = (np.ones((len(t), 1)) * weights * IRF.T)
    
    # Normalize by sum of all components
    normalization = np.ones((len(t), 1)) @ (np.sum(IRF, axis=1, keepdims=True).T)
    IRF = IRF_weighted / normalization

    # Clip extreme values to prevent numerical issues
    tm = 0.5 * np.max(IRF[:, 0])  # Half-maximum of main component
    IRF[IRF[:, 1] > tm, 1] = tm   # Clip tail component
    IRF[IRF[:, 2] > tm, 2] = tm   # Clip reconvolution component

    # Handle NaN values and ensure non-negativity
    IRF[np.isnan(IRF)] = 0
    IRF = (np.sum(IRF, axis=1) / np.sum(np.sum(IRF)))[:, np.newaxis]
    IRF[IRF < 0] = 0

    # Ensure monotonic rise before peak to maintain causality
    t0 = np.argmax(IRF)  # Find peak position
    if t0 > 0:
        tmp = IRF[:t0, 0]  # Values before peak
        tmp = np.diff(tmp)  # Compute differences
        tmp[tmp < 0] = 0    # Remove negative slopes
        tmp = np.concatenate(([IRF[0, 0]], np.cumsum(tmp)))  # Reconstruct monotonic rise
        IRF[:t0, 0] = tmp

    # Final normalization
    z = np.sum(IRF, axis=1) / np.sum(np.sum(IRF))

    # Optional plotting
    if pic is not None:
        if pic == 1:
            plt.figure()
            plt.plot(t, z, 'r', linewidth=2, label='IRF Model')
            plt.xlabel('Time')
            plt.ylabel('IRF Amplitude')
            plt.title('Instrumental Response Function (Linear Scale)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.figure()
            plt.semilogy(t, z, 'r', linewidth=2, label='IRF Model')
            plt.xlabel('Time')
            plt.ylabel('IRF Amplitude (log)')
            plt.title('Instrumental Response Function (Log Scale)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.draw()

    return z


def TCSPC_Fun(p, t, y=None, para=None):
    """
    Fits multi-exponential decay model to Time-Correlated Single Photon Counting (TCSPC) data.
    
    This function implements a comprehensive TCSPC fitting model that convolves an
    instrumental response function (IRF) with exponential decay functions. It can
    operate in two modes:
    1. Forward mode: Generate model curves from parameters
    2. Fitting mode: Fit model to experimental data using non-negative least squares
    
    The mathematical model is:
    y(t) = c₀ + Σᵢ cᵢ * Convol(IRF(t), exp(-(t-t₀)/τᵢ))
    
    Where:
    - c₀ is a constant background term
    - cᵢ are the amplitudes for each exponential component
    - τᵢ are the decay time constants
    - IRF(t) is the instrumental response function
    
    Parameters
    ----------
    p : array-like, shape (7 + n_exponentials,)
        Model parameters where:
        p[0:7] = IRF parameters [t₀, w1, T1, T2, a, b, dt]
        p[7:] = decay times [τ₁, τ₂, ..., τₙ]
    t : array-like
        Time vector or coefficient matrix (context-dependent)
    y : array-like, optional
        Experimental decay data to fit. If None, function operates in forward mode.
        Can be 1D (single curve) or 2D (multiple curves)
    para : array-like, optional
        Fixed IRF parameters. If provided, these replace corresponding entries in p
        
    Returns
    -------
    err : float
        Sum of squared residuals between data and fit
    c : ndarray
        Fitted coefficients (amplitudes) for each component
    zz : ndarray, shape (n_time, n_components+1)
        Model matrix containing basis functions:
        - Column 0: constant term (background)
        - Column i+1: convolved exponential component i
    z : ndarray
        Fitted curves (zz @ c)
        
    Notes
    -----
    - Uses non-negative least squares (NNLS) to ensure physical constraints
    - Handles NaN/Inf values gracefully with error reporting
    - IRF is computed using IRF_Fun() with first 7 parameters
    - Convolution performed using Convol() function
    
    Examples
    --------
    >>> # Generate synthetic TCSPC data
    >>> p = [0, 0.1, 0.05, 0.1, 0.01, 0.1, 0, 1.0, 3.0]  # IRF + 2 lifetimes
    >>> t = np.linspace(0, 10, 100)
    >>> err, c, zz, z = TCSPC_Fun(p, None, t)  # Forward mode
    
    >>> # Fit experimental data
    >>> y_data = np.random.poisson(z) + 1  # Add noise
    >>> err, c, zz, z_fit = TCSPC_Fun(p, t, y_data)  # Fitting mode
    """
    
    # Ensure input parameters are proper numpy arrays
    p = np.array(p).flatten()

    # Handle parameter substitution if fixed parameters are provided
    if para is not None:
        para = np.array(para).flatten()
        n = len(para)
        if n > 6:
            # Replace all IRF parameters with provided ones
            p = np.concatenate([para, p])
        else:
            # Insert fixed parameters at appropriate positions in IRF parameter set
            p = np.concatenate([p[:7-n], para, p[7-n:]])

    # FORWARD MODE: Generate model curves from coefficients
    if y is None or len(t) < len(y):
        # In this mode, t contains coefficients and y contains time vector
        c = t  # Coefficient matrix
        t = np.array(y).flatten()  # Time vector
        
        # Extract decay parameters
        nex = len(p) - 7  # Number of exponential components
        tauT = p[7:]      # Decay time constants
        
        # Generate IRF from first 7 parameters
        IRF = IRF_Fun(p[:7], t)
        
        # Build model matrix: constant + convolved exponentials
        zz = np.zeros((len(t), nex + 1))
        zz[:, 0] = 1  # Constant background term
        
        # Generate each exponential component convolved with IRF
        for i in range(nex):
            # Create normalized exponential decay starting at t₀
            decay = np.exp(-(t - p[0]) / tauT[i]) / tauT[i]
            # Convolve with IRF to account for instrument response
            tmp = Convol(IRF, decay)
            zz[:, i+1] = tmp[:len(t)]
        
        # Apply coefficients to generate multiple curves
        err = np.zeros((len(t), c.shape[1]))
        z = np.zeros((len(t), c.shape[1]))  # Initialize z for forward mode
        for j in range(c.shape[1]):
            err[:, j] = zz @ c[:, j]  # Linear combination of basis functions
            z[:, j] = err[:, j]  # Copy to z for consistency

    # FITTING MODE: Fit model to experimental data
    else:
        # Prepare input arrays
        t = np.array(t).flatten()
        y = np.array(y)
        
        # Ensure y is in correct orientation (time × curves)
        m, n = y.shape
        if m < n:
            y = y.T
            m, n = y.shape
        
        # Remove time points with invalid data (NaN, Inf)
        valid_idx = np.isfinite(np.sum(y, axis=1))
        t = t[valid_idx]
        y = y[valid_idx, :]
        
        # Extract fitting parameters
        nex = len(p) - 7  # Number of exponential components
        IRF = IRF_Fun(p[:7], t)  # Generate IRF
        tauT = p[7:]      # Decay time constants
        t1 = t - p[0]     # Time relative to IRF peak
        
        # Build model matrix (design matrix for linear regression)
        zz = np.zeros((len(t), nex + 1))
        zz[:, 0] = 1  # Constant background term
        
        # Generate convolved exponential basis functions
        for i in range(nex):
            # Normalized exponential decay function
            decay = np.exp(-t1 / tauT[i]) / tauT[i]
            # Convolve with IRF
            tmp = Convol(IRF, decay)
            zz[:, i+1] = tmp[:len(t)]
        
        # Initialize output arrays
        c = np.zeros((zz.shape[1], n))  # Fitted coefficients
        z = np.zeros_like(y)            # Fitted curves
        
        # Fit each curve using non-negative least squares
        for j in range(n):
            try:
                # Check for valid data (no NaN/Inf)
                if not (np.isnan(np.sum(zz)) or np.isnan(np.sum(y[:,j]))):
                    # Non-negative least squares fitting (ensures physical constraints)
                    c[:, j], _ = nnls(zz, y[:, j])
                    z[:, j] = zz @ c[:, j]  # Reconstruct fitted curve
                else:
                    # Handle invalid data gracefully
                    c[:,j] = 0
                    z[:, j] = 0
            except Exception as e:
                # Catch any fitting errors and continue with zeros
                print(f"Fitting error for curve {j}: {e}")
                c[:,j] = 0
                z[:, j] = 0
        
        # Calculate sum of squared residuals as error metric
        err = np.sum((y - z)**2)
    
    return err, c, zz, z


def Calc_mIRF(head, tcspc):
    """
    Calculate the measured Instrumental Response Function (mIRF) for TCSPC data.
    
    This function estimates the IRF for each pixel/channel in TCSPC data by fitting
    a multi-component IRF model to the measured decay curves. It uses a cascaded
    optimization approach with random parameter initialization to find robust fits.
    
    The process involves:
    1. Initial IRF parameter estimation from the brightest pixel
    2. Individual fitting for each pixel using fixed IRF shape parameters
    3. Robust optimization with multiple random initializations
    
    Parameters
    ----------
    head : dict
        Metadata dictionary containing:
        - 'MeasDesc_Resolution': Time resolution per channel (seconds)
        - Additional measurement parameters
    tcspc : ndarray, shape (n_pixels, n_time_bins, n_pie_channels)
        3D TCSPC data array where:
        - Axis 0: Spatial pixels/channels
        - Axis 1: Time bins (histogram channels)
        - Axis 2: PIE (Pulsed Interleaved Excitation) channels
    
    Returns
    -------
    IRF : ndarray, shape (n_pixels, n_time_bins, n_pie_channels)
        Calculated IRF array with same dimensions as input tcspc data.
        Each IRF is normalized and represents the instrument response
        for the corresponding pixel and excitation channel.
    
    Notes
    -----
    - Uses IRF_Fun() for modeling with 7 shape parameters plus decay times
    - Employs cascaded optimization with 10 iterations of random restarts
    - Initial parameters are based on peak position analysis
    - Non-negative constraints ensure physical validity
    - Error handling prevents NaN/Inf values in output
    
    Algorithm Details
    ---------------
    1. Extract time resolution and create time axis
    2. Find peak positions to estimate t₀ parameter
    3. Fit brightest pixel first to get global IRF parameters
    4. Use fixed IRF shape for all other pixels (varies only t₀ and amplitudes)
    5. Multiple random initializations ensure robust convergence
    
    Examples
    --------
    >>> # Typical usage with PTU file data
    >>> head = {'MeasDesc_Resolution': [4e-12]}  # 4 ps resolution
    >>> tcspc_data = np.random.poisson(100, (64, 64, 256, 2))  # 64x64 pixels, 256 time bins
    >>> irf_measured = Calc_mIRF(head, tcspc_data)
    """

    # Temporarily suppress numerical warnings during optimization
    np.seterr(divide='ignore', over='ignore', invalid='ignore')
    
    # Extract time resolution and convert to nanoseconds
    maxres = np.max(head['MeasDesc_Resolution']*1e9)
    Resolution = max([maxres, 0.1])  # Minimum 0.1 ns resolution
    # Pulse = 1e9 / head['SyncRate']  # Laser repetition period (if needed)

    # Create time axis: bin centers in nanoseconds
    tau = Resolution * (np.arange(tcspc.shape[1]) + 0.5)
    IRF = np.zeros(tcspc.shape)  # Initialize output IRF array
    nex = 2  # Number of exponential decay components for initial fitting

    # Estimate peak position from maximum of each decay curve
    t0_idx = np.argmax(tcspc, axis=1)
    t0 = tau[min(t0_idx.min(), len(tau) - 1)]  # Global peak position estimate

    # Initial IRF parameter estimates (empirically determined)
    w1 = 0.03**2      # Gaussian width parameter (squared)
    T1 = 0.050        # First time constant (ns)
    T2 = 0.10         # Second time constant (ns)
    a = 0.005         # Reconvolution amplitude
    b = 0.1           # Tail amplitude
    dt = 0.0          # Time delay parameter

    # Process each PIE (excitation) channel separately
    for PIE in range(tcspc.shape[2]):

        # Initial parameter vector: [IRF params (7) + decay times (2)]
        p = np.array([t0, w1, T1, T2, a, b, dt, 0.5, 2])
        # Parameter bounds: lower limits
        pl = np.array([t0 - 2.5, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, -0.3] + [0.1] * nex)
        # Parameter bounds: upper limits
        pu = np.array([t0 + 2.5, 1, 1, 1, 0.01, 0.5, 0.5] + [10] * nex)

        # Sort pixels by total counts (brightest first)
        tc = np.sum(tcspc[:, :, PIE], axis=1)
        ord = np.argsort(tc)[::-1]

        # Start with the brightest pixel for initial IRF parameter estimation
        ch = 0
        ind = ord[ch]
        y = tcspc[ind, :, PIE][:,np.newaxis]

        # Arrays for storing optimization results
        err = np.zeros(10)  # Error values for each sub-iteration
        p_array = np.zeros((len(p), 10))  # Parameter sets

        # Cascaded optimization with random restart strategy
        for casc in range(10):
            if casc == 0:
                r0 = p  # Use initial guess for first cascade
            else:
                # Use best result from previous cascade as starting point
                s = err.argmin()
                r0 = p_array[:, s]
                
            # Multiple random initializations within each cascade level
            for sub in range(10):
                # Generate random perturbation (decreasing with cascade level)
                rf = r0 * (2 ** (1.1 * (np.random.rand(len(r0)) - 0.5) / (casc+1)))
                rf = np.clip(rf, pl, pu)  # Enforce parameter bounds
                
                # Perform optimization
                res = minimize_s(lambda x: TCSPC_Fun(x, tau, y.astype(np.float64))[0], rf, bounds=list(zip(pl, pu)))
                p_array[:, sub] = res.x
                err[sub] = TCSPC_Fun(p_array[:, sub], tau, y.astype(np.float64))[0]

        # Select best result and compute IRF for brightest pixel
        err1 = err.min()
        p1 = np.mean(p_array[:, err == err1], axis=1)  # Average best solutions
        _, c1, _, tmp1 = TCSPC_Fun(p1, tau, y.astype(np.float64))
        IRF[ind, :, PIE] = IRF_Fun(p1[:7], tau)

        # Fix IRF shape parameters, vary only timing and decay amplitudes
        para = p1[1:7]  # Fixed IRF shape parameters (w1, T1, T2, a, b, dt)
        p = np.concatenate([[p1[0]], p1[7:]])  # Variable parameters: [t0, decay_times]
        pl = np.array([0] + [.1] * nex)   # New lower bounds
        pu = np.array([3] + [10] * nex)   # New upper bounds

        for ch in range(0, tcspc.shape[0]):
            ind = ord[ch]
            y = tcspc[ind, :, PIE][:,np.newaxis]
            # Debug: print(len(y))
            err = np.zeros(10)
            p_array = np.zeros((len(p), 10))

            for casc in range(10):
                if casc==0:
                    r0 = p
                else:
                    s = err.argmin()
                    r0 = p_array[:, s]
                for sub in range(10):
                    rf = r0 * (2 ** (1.05 * (np.random.rand(len(r0)) - 0.5) / (casc+1)))
                    rf = np.clip(rf, pl, pu)
                    res = minimize_s(lambda x: TCSPC_Fun(x, tau, y.astype(np.float64), para)[0], rf, bounds=list(zip(pl, pu)))
                    p_array[:, sub] = res.x
                    err[sub] = TCSPC_Fun(p_array[:, sub], tau, y.astype(np.float64), para)[0]

            err1 = err.min()
            p1 = np.mean(p_array[:, err == err1], axis=1)
            _, c1, _, tmp1 = TCSPC_Fun(p1, tau, y.astype(np.float64), para)


            IRF[ind, :, PIE] = IRF_Fun(np.concatenate([[p1[0]], para, p1[1:]]), tau)

    IRF[IRF < 0] = 0
    np.seterr(divide='warn', over='warn', invalid='warn')
    
    return IRF


def PIRLSnonneg_batch(M, Y, max_num_iter=10):
    """
    Batch PIRLS solver across multiple right-hand sides (pixels) without using Parallel.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.
    Y : ndarray, shape (n_samples, n_pixels)
        Observation matrix, one column per pixel.
    max_num_iter : int, optional
        Number of PIRLS iterations.

    Returns
    -------
    Beta : ndarray, shape (n_features, n_pixels)
        Fitted non-negative coefficients per pixel.
    """
    n_pixels = Y.shape[1]
    # Beta = np.zeros((X.shape[1], n_pixels), dtype=float)
    lr = LinearRegression(positive=True, fit_intercept=False)
    lr.fit(M, Y)
    # lr.coef_ is (n_targets, n_features) = (npix, n_basis)
    Beta= lr.coef_.T
    # Flatten pixel dimension: operate on each column sequentially
    # but reshaped 2D->1D pixel axis
    for i in tqdm(range(n_pixels),desc = 'Fitting amplitudes for pixel/window'):
        beta, _ = nnls(M, Y[:, i])
        n = M.shape[0]
        TINY = 0.1 / n
        for _ in range(max_num_iter):
            w = 1.0 / np.maximum(M.dot(beta), TINY)
            MtW = M.T * w[np.newaxis, :]
            Aw = MtW.dot(M)
            bw = MtW.dot(Y[:, i])
            beta_new, _ = nnls(Aw, bw)
            if np.linalg.norm(beta_new - beta) < 1e-10:
                beta = beta_new
                break
            beta = beta_new
        Beta[:, i] = beta
    return Beta


def PIRLSnonneg_batch_gpu(M, Y, max_num_iter=10):
    """
    GPU-accelerated batch PIRLS solver with fallback to CPU implementation.
    
    This function automatically tries to use GPU acceleration when available,
    falling back to CPU implementation if GPU is not available or fails.

    Parameters
    ----------
    M : ndarray, shape (n_samples, n_features)
        Design matrix.
    Y : ndarray, shape (n_samples, n_pixels)
        Observation matrix, one column per pixel.
    max_num_iter : int, optional
        Number of PIRLS iterations.

    Returns
    -------
    Beta : ndarray, shape (n_features, n_pixels)
        Fitted non-negative coefficients per pixel.
    """
    
    # Try GPU implementation first
    if CUDA_AVAILABLE and pirls_batch_cuda is not None:
        try:
            n_samples, n_basis = M.shape
            n_pixels = Y.shape[1]
            
            # Check if problem size fits GPU constraints
            if n_basis <= 64 and n_samples <= 2048:
                print(f" Using GPU acceleration for PIRLS fitting ({n_pixels} pixels)")
                Beta = pirls_batch_cuda(M, Y, max_num_iter)
                return Beta
            else:
                print(f"  Problem size too large for GPU (n_basis={n_basis}, n_samples={n_samples})")
                print("   Falling back to CPU implementation...")
        except Exception as e:
            print(f"  GPU PIRLS failed: {e}")
            print("   Falling back to CPU implementation...")
    
    # Fallback to CPU implementation
    print(f" Using CPU implementation for PIRLS fitting ({Y.shape[1]} pixels)")
    return PIRLSnonneg_batch(M, Y, max_num_iter)


def PIRLSnonneg(x, y, max_num_iter=10):
    """
    Poisson Iterative Reweighted Least Squares (PIRLS) for non-negative coefficients.
    
    This function solves the weighted least squares problem X * β = Y for Poisson-
    distributed data Y with the constraint that β ≥ 0. It uses an iterative
    reweighting scheme where weights are updated based on the current fit to
    account for the heteroscedastic nature of Poisson noise.
    
    The algorithm iteratively solves:
    β^(k+1) = argmin ||W^(1/2)(k) (Xβ - Y)||² subject to β ≥ 0
    
    Where W(k) is a diagonal weight matrix with W_ii = 1/max(Xβ^(k), ε)
    
    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        Design matrix (predictor/independent variables).
        Each row represents an observation, each column a feature.
    y : array-like, shape (n_samples,)
        Response vector (dependent variable, assumed Poisson-distributed).
        Must be non-negative as expected for count data.
    max_num_iter : int, optional, default=10
        Maximum number of PIRLS iterations to perform.
        Algorithm may converge earlier if tolerance is met.
    
    Returns
    -------
    beta : ndarray, shape (n_features,)
        Solution vector with non-negative coefficients.
        Represents the fitted parameters of the linear model.
    k : int
        Actual number of iterations performed before convergence
        or reaching maximum iterations.
    
    Notes
    -----
    - Convergence is determined by ||β^(k+1) - β^(k)||² < 1e-10
    - Uses NNLS (Non-Negative Least Squares) for constrained optimization
    - Regularization parameter TINY prevents division by zero
    - Suitable for TCSPC data where Poisson statistics dominate
    
    Algorithm Steps
    ---------------
    1. Initialize β using standard NNLS
    2. For each iteration:
       a. Update weights W_ii = 1/max(Xβ, TINY)
       b. Solve weighted NNLS: min ||W^(1/2)(Xβ - Y)||² s.t. β ≥ 0
       c. Check convergence criterion
    3. Return final β and iteration count
    
    Examples
    --------
    >>> # Simple example with synthetic Poisson data
    >>> X = np.random.randn(100, 5)
    >>> beta_true = np.array([1, 0, 2, 0, 0.5])
    >>> y_mean = X @ beta_true
    >>> y = np.random.poisson(np.exp(y_mean))  # Poisson data
    >>> beta_fit, n_iter = PIRLSnonneg(X, y, max_num_iter=20)
    """
    
    n = len(y)
    TINY = 0.1 / n  # Small regularization value
    w = np.zeros((n, n))
    
    # Initial guess using non-negative least squares
    # beta_last = lsq_linear(x, y, bounds=(0, np.inf)).x
    beta_last,_ = nnls(x,y)
    beta = beta_last.copy()  # Initialize beta to ensure it's always defined
    k = 0  # Initialize k to ensure it's always defined
    
    for k in range(max_num_iter):
        # Update the weight matrix with regularization
        w[np.diag_indices_from(w)] = 1. / np.maximum(x @ beta_last, TINY)
        
        # Update beta - maximum likelihood solution vector
        xt_w = x.T @ w
        # beta = lsq_linear(xt_w @ x, xt_w @ y, bounds=(0, np.inf)).x
        beta,_ = nnls(xt_w @ x, xt_w @ y)
        
        # Check for convergence
        delta = beta - beta_last
        if np.sum(delta ** 2) < 1e-10:
            break
        
        beta_last = beta
    
    return beta, k

def MLFit(param, y, irf, p, plt_flag = None):
    """
    Computes Maximum Likelihood error for fluorescence decay fitting with periodic excitation.
    
    This function implements a maximum likelihood estimator for fitting multi-exponential
    fluorescence decay data. It accounts for periodic laser excitation by incorporating
    the time between excitation pulses and uses color shift correction to align the IRF
    with the decay data.
    
    The mathematical model assumes:
    y(t) = A₀ + Σᵢ Aᵢ * Convol(IRF_shifted(t), exp(-t/τᵢ) / (1-exp(-p/τᵢ)))
    
    Where the normalization factor (1-exp(-p/τᵢ)) accounts for periodic excitation.
    
    Parameters
    ----------
    param : array-like, shape (1 + n_components,)
        Fitting parameters where:
        param[0] : float
            Color shift between IRF and decay data (in time channels).
            Positive values shift IRF forward in time.
        param[1:] : array-like
            Decay time constants τᵢ for each exponential component.
    y : array-like, shape (n_time_bins,)
        Measured fluorescence decay curve (photon counts per time bin).
        Should be non-negative integers for proper Poisson statistics.
    irf : array-like, shape (n_time_bins,)
        Measured Instrumental Response Function.
        Represents the system response to an instantaneous pulse.
    p : float
        Time between laser excitation pulses (in TCSPC time channels).
        Used to correct for periodic excitation effects.
    plt_flag : bool, optional
        Plotting flag (currently not implemented).
        Reserved for future visualization capabilities.

    Returns
    -------
    err : float
        Maximum likelihood error (negative log-likelihood normalized by degrees of freedom).
        Lower values indicate better fits. Formula:
        err = Σᵢ [y[i]*log(y[i]/z[i]) - y[i] + z[i]] / (n - n_params)
        
    Notes
    -----
    - Uses linear interpolation for sub-channel color shift correction
    - Employs non-negative least squares to ensure physical constraints
    - Includes constant background term in the fit
    - Handles periodic boundary conditions through modulo operations
    - Numerical stability ensured by excluding zero-count time bins
    
    Algorithm Steps
    ---------------
    1. Extract color shift and decay times from parameters
    2. Generate periodic decay functions accounting for pulse repetition
    3. Apply sub-channel color shift to IRF using interpolation
    4. Convolve shifted IRF with each decay component
    5. Fit amplitudes using non-negative least squares
    6. Calculate maximum likelihood error on valid data points
    
    Examples
    --------
    >>> # Synthetic data example
    >>> irf = np.exp(-np.arange(100)**2/50)  # Gaussian IRF
    >>> tau_true = [1.0, 5.0]  # True decay times
    >>> param_guess = [0.5] + tau_true  # Initial guess with color shift
    >>> y_data = generate_decay_data(irf, tau_true, noise=True)
    >>> error = MLFit(param_guess, y_data, irf, p=100)
    """
    
    n = len(irf)
    t = np.arange(1, n+1)
    tp = np.arange(1, p+1)
    c = param[0]
    tau = np.array(param[1:])
    
    # Matrix x calculation
    x = np.exp(-(tp[:, None] - 1) * (1.0 / tau)) @ np.diag(1.0 / (1 - np.exp(-p / tau)))
    irs = (1 - c + np.floor(c)) * irf[(t - np.int_(c) - 1) % n] + \
         (c - np.floor(c)) * irf[(t - int(np.ceil(c)) - 1) % n]
    
    # Perform the convolution using the Convol function
    z = Convol(irs, x)
    
    # Add a constant term
    # z = np.hstack([np.ones((z.shape[0], 1)), z])
    z = np.column_stack((np.ones(len(z)), z))
   
    # Perform non-negative least squares to fit A
    A = lsq_linear(z, y, bounds=(0, np.inf)).x
    # A,_ = PIRLSnonneg(z,y, 100) 
    # A,_ = nnls(z,y)
    # 
    # Recompute z using the estimated coefficients A
    z = z @ A
    
    # Calculate the error using Maximum Likelihood approach
    ind = y > 0
    err = np.sum(y[ind] * np.log(y[ind] / z[ind]) - y[ind] + z[ind]) / (n - len(tau))
    
    # Debug: print(err)
    return err

def LSFit(param, y, irf, p, plt_flag = None):
    """
    Computes Least-Squares deviation for fluorescence decay fitting with periodic excitation.
    
    This function implements a least squares estimator for fitting multi-exponential
    fluorescence decay data, accounting for periodic laser excitation. It uses the same
    mathematical model as MLFit but with a different error metric (currently using
    negative log-likelihood despite the name).
    
    The fitted model assumes:
    y(t) = A₀ + Σᵢ Aᵢ * Convol(IRF_shifted(t), exp(-t/τᵢ) / (1-exp(-p/τᵢ)))
    
    Where the normalization factor accounts for finite pulse repetition period.
    
    Parameters
    ----------
    param : array-like, shape (1 + n_components,)
        Fitting parameters where:
        param[0] : float
            Color shift between IRF and decay data (in time channels).
            Positive values delay the IRF relative to the decay.
        param[1:] : array-like
            Decay time constants τᵢ for each exponential component.
    y : array-like, shape (n_time_bins,)
        Measured fluorescence decay curve (photon counts per time bin).
        Should contain non-negative values representing detected photons.
    irf : array-like, shape (n_time_bins,)
        Measured Instrumental Response Function.
        Represents system response to delta function excitation.
    p : float
        Time between laser excitation pulses (in TCSPC time channels).
        Accounts for periodic boundary conditions in decay fitting.
    plt_flag : bool, optional
        If not None, displays real-time fitting progress plots.
        Shows experimental data and current fit on semi-log scale.

    Returns
    -------
    err : float
        Fitting error metric. Despite the function name suggesting least squares,
        this currently implements negative log-likelihood:
        err = Σᵢ [y[i]*log(y[i]/z[i]) - y[i] + z[i]] / (n - n_params)
        
    Notes
    -----
    - Color shift correction uses linear interpolation between time bins
    - Non-negative least squares ensures physical constraints (A ≥ 0)
    - Background offset automatically included in fit
    - Real-time plotting available for monitoring convergence
    - Handles periodic boundary conditions via modulo arithmetic
    
    Implementation Details
    ----------------------
    1. Extract color shift parameter and decay times
    2. Generate exponential basis functions with periodic normalization
    3. Apply fractional color shift to IRF using interpolation
    4. Convolve shifted IRF with each decay basis function
    5. Solve for amplitudes using constrained least squares
    6. Calculate error metric on positive-count bins only
    
    Examples
    --------
    >>> # Example with synthetic noisy decay data
    >>> irf = gaussian_irf(width=0.1, length=256)
    >>> y_data = synthetic_decay([2.0, 8.0], amplitudes=[0.3, 0.7])
    >>> param_init = [0.0, 2.5, 7.5]  # [color_shift, tau1, tau2]
    >>> error = LSFit(param_init, y_data, irf, p=256, plt_flag=True)
    
    See Also
    --------
    MLFit : Maximum likelihood version of the same fitting problem
    FluoFit : High-level interface for fluorescence lifetime fitting
    """
    n = len(irf)
    t = np.arange(1, n+1)
    tp = np.arange(1, p+1)
    c = param[0]
    tau = np.array(param[1:])
    
    # Matrix x calculation
    x = np.exp(-(tp[:, None] - 1) * (1.0 / tau)) @ np.diag(1.0 / (1 - np.exp(-p / tau)))
    irs = (1 - c + np.floor(c)) * irf[(t - np.int_(c) - 1) % n] + \
         (c - np.floor(c)) * irf[(t - int(np.ceil(c)) - 1) % n]
         
    z = Convol(irs, x)
    # Add column of ones to z for fitting
    z = np.column_stack((np.ones(len(z)), z))
    # Linear least squares solution for A
    A = lsq_linear(z, y, bounds=(0, np.inf)).x
    # A,_ = nnls(z, y)
    # A,_ = PIRLSnonneg(z,y,10)
    # print(A.shape)
    # Generate fitted curve
    z = z @ A
    
    if plt_flag is not None:
        # plt.semilogy(t, irs / np.max(irs) * np.max(y), label="irs")
        plt.semilogy(t, y, 'bo', label="y")
        plt.semilogy(t, z, label="fitted z")
        plt.legend()
        plt.draw()
        plt.pause(0.001)
        
        
    # Error calculation (Least-squares deviation)
    # TINY = 10**-10
    # err = np.sum((z+TINY - y) ** 2 / np.abs(z+TINY)) / (n - len(tau))
    ind = y > 0
    err = np.sum(y[ind] * np.log(y[ind] / z[ind]) - y[ind] + z[ind]) / (n - len(tau))
    
    # Debug: print(err)
    return err

def DistFluoFit( y, p, dt, irf=None, shift=(-10,10), flag=0, bild = None, N = 100, scattering = True):
    """
    Performs distributed fluorescence lifetime fitting using regularized linear regression.
    
    This function estimates a continuous distribution of fluorescence lifetimes rather than
    fitting discrete exponential components. It uses a basis set of exponentially spaced
    lifetime values and solves for their relative amplitudes using non-negative least squares.
    The method is particularly useful for samples with heterogeneous fluorophore populations
    or complex decay kinetics.
    
    Key features:
    - Color shift optimization for IRF alignment
    - Optional scattered light component inclusion
    - Lifetime distribution visualization
    - Robust handling of different shift ranges
    
    Parameters
    ----------
    y : array-like, shape (n_time_bins,)
        Fluorescence decay data (photon counts per time bin).
        Should be background-subtracted and contain sufficient signal.
    p : float
        Time between laser excitation pulses (nanoseconds).
        Defines the measurement window and repetition period.
    dt : float
        Time width of one TCSPC channel (nanoseconds).
        Determines time resolution and conversion between bins and time.
    irf : array-like, optional, shape (n_time_bins,)
        Instrumental Response Function. If None, assumes delta function
        (sets first time bin to 1, rest to 0).
    shift : tuple or scalar, optional, default=(-10, 10)
        Color shift search range (TCSPC channels).
        - If tuple: (min_shift, max_shift) boundaries
        - If scalar: symmetric range [-|shift|, |shift|]
    flag : int, optional, default=0
        Post-processing flag for lifetime binning:
        - 0: Return full distribution
        - >0: Bin significant lifetime components
    bild : int, optional
        Plotting flag for visualization:
        - None: No plots
        - Any value: Generate diagnostic plots
    N : int, optional, default=100
        Number of trial lifetime values for distribution estimation.
        More values give finer resolution but increase computation time.
    scattering : bool, optional, default=True
        Whether to include scattered light (IRF) component in the fit.
        Useful for samples with significant Rayleigh/Raman scattering.
        
    Returns
    -------
    cx : ndarray, shape (N+1,) or (n_binned,)
        Lifetime distribution amplitudes (normalized).
        If flag=0: full distribution; if flag>0: binned components.
    tau : ndarray, shape (N+1,) or (n_binned,)
        Corresponding lifetime values (nanoseconds).
        Exponentially spaced from fast to slow components.
    offset : float
        Background offset level from the fit.
    csh : float
        Optimal color shift (TCSPC channels) for IRF alignment.
        Positive values indicate IRF is delayed relative to decay.
    z : ndarray, shape (n_time_bins,)
        Fitted fluorescence curve using optimal parameters.
    t : ndarray, shape (n_time_bins,)
        Time axis corresponding to z (TCSPC channel indices).
    err : float
        Fitting error (weighted chi-squared metric).
        
    Notes
    -----
    - Uses exponentially spaced lifetime grid for uniform log coverage
    - Color shift optimization performed over specified range
    - All decay functions normalized to unit area before fitting
    - Includes periodic boundary condition handling
    - Regularization prevents overfitting to noise
    
    Algorithm Steps
    ---------------
    1. Initialize IRF if not provided
    2. Generate exponentially-spaced lifetime basis set
    3. Create normalized decay model matrix
    4. Optimize color shift by grid search
    5. Solve for amplitudes using NNLS
    6. Optionally bin significant lifetime components
    7. Generate diagnostic plots if requested
    
    Examples
    --------
    >>> # Basic distributed lifetime analysis
    >>> irf_measured = load_irf_data()
    >>> decay_data = load_decay_data()
    >>> cx, tau, offset, shift, fit, t, error = DistFluoFit(
    ...     decay_data, p=12.5, dt=0.032, irf=irf_measured)
    >>> plot_lifetime_distribution(tau, cx)
    
    >>> # High-resolution analysis with plotting
    >>> cx, tau, offset, shift, fit, t, error = DistFluoFit(
    ...     decay_data, p=12.5, dt=0.032, N=200, bild=1)
    
    See Also
    --------
    FluoFit : Discrete multi-exponential fitting
    TCSPC_Fun : Low-level TCSPC model function
    """
    if irf is None:
        irf = np.zeros(int(p/dt))
        irf[0] = 1  # just put 1 in the first time bin for IRF
    irf = np.array(irf).flatten()
    y   = np.array(y).flatten()
    n = len(irf)
    tp = dt*np.arange(p/dt) 
    t = np.arange(1,n+1)
    if hasattr(shift, '__len__') and len(shift) == 1:
        shmin = -np.abs(shift[0]).astype(np.int32)
        shmax = np.abs(shift[0]).astype(np.int32)
    elif hasattr(shift, '__len__') and len(shift) == 2:
        shmin, shmax = shift
    elif not hasattr(shift, '__len__'):
        # shift is a scalar
        shmin = -np.abs(shift).astype(np.int32)
        shmax = np.abs(shift).astype(np.int32)
    else:
        print('shift should have only one or two values, sh_min and/or sh_max')
        shmin, shmax = -10, 10  # default values
        
    tau = (1/dt)/np.exp(np.arange(N+1)/N * np.log(p/dt)) # distribution of decay rates (inverse of lifetimes)
    if scattering is True:
        M0 = np.column_stack((np.ones(len(t)), irf, Convol(irf,np.exp(-tp[:, None]*tau))))
    else:  
        M0 = np.column_stack((np.ones(len(t)), Convol(irf,np.exp(-tp[:, None]*tau))))
    M0 = M0/(np.ones(n)[:, None]*np.sum(M0,axis=0)); # Normalized decay curves
    
    # search for optimal irf colorshift   
    err = np.empty(0)     
    TINY = 10**-30
    if shmax-shmin>0:
        for c in range(shmin,shmax+1):
            M = (1 - c + np.floor(c)) * M0[(t - np.int_(c) - 1) % n,:] + \
                 (c - np.floor(c)) * M0[(t - int(np.ceil(c)) - 1) % n,:]
            ind = np.arange(np.max([0,c]), np.min([n-1,n+c-1])).astype(np.int32)     
            # cx,_ = PIRLSnonneg(M[ind,: ],y[ind],10)
            cx,_ = nnls(M[ind,:],y[ind])
            z = M @ cx
            err = np.concatenate((err, [np.sum((z-y+TINY)**2/np.abs(z+TINY)/len(ind))]))
        shv = np.arange(shmin,shmax+0.1,0.1)
        tmp = np.interp(shv,np.arange(shmin,shmax+1),err)
        pos = np.argmin(tmp)
        csh = shv[pos]
    else:
        csh = shmin
        
    M = (1 - csh + np.int_(csh)) * M0[(t - np.int_(csh) - 1) % n,:] + \
          (csh - np.int_(csh)) * M0[(t - int(np.ceil(csh)) - 1) % n,:]
    c = np.ceil(np.abs(csh))*np.sign(csh)
    ind = np.arange(np.max([0,c]),np.min([n,n+c])).astype(np.int32) 
    # cx,_ = PIRLSnonneg(M[ind,:],y[ind])
    cx,_ = nnls(M[ind,:],y[ind])
    z = M @ cx
    err = np.sum((z-y+TINY)**2/np.abs(z+TINY)/len(ind))
    
    if bild is not None:
        t = dt * t # tau 
        
        # First plot: semilogarithmic plot of y and z
        plt.figure()
        plt.semilogy(t, y, 'ob', linewidth=1, label='y data')
        plt.semilogy(t, z, 'r', linewidth=2, label='z data')
        plt.xlabel('time [ns]')
        plt.ylabel('lg count')
        
        # Adjusting axis limits
        v = [min(t), max(t)]
        plt.xlim(v[0], v[1])
        plt.legend()
        plt.show()
        
        # Second figure: Weighted residuals
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, (y - z) / np.sqrt(z))
        plt.xlabel('time [ns]')
        plt.ylabel('weighted residual')
        
        # Adjust axis limits
        v = [min(t), max(t)]
        plt.xlim(v[0], v[1])
        
        # Calculate fac and tau for next plot
        len_ind = len(cx) - 2
        ind = np.arange(len_ind)
        tau = 1.0 / tau  # Reciprocal of tau
        fac = np.sqrt(np.dot(tau[:-1], 1.0/tau[1:]))

        
        # Subplot for distribution
        plt.subplot(2, 1, 2)
        # x_vals = np.reshape([fac * tau[ind], fac * tau[ind], tau[ind] / fac, tau[ind]], (4 * len_ind, 1))
        # y_vals = np.reshape([0 * tau[ind], cx[ind + 1], cx[ind + 1], 0 * tau[ind]], (4 * len_ind, 1))
        
        # # Semilogarithmic plot with patch-like behavior
        # plt.semilogx(x_vals, y_vals)
        # plt.fill_between(x_vals.flatten(), y_vals.flatten(), color='b', alpha=0.3)
        
        plt.semilogx(tau[ind],cx[ind+1])
        plt.xlabel('decay time [ns]')
        plt.ylabel('distribution')
        plt.show()
        
        t = t/dt
        tau = 1.0/tau
     

    offset = cx[0]
    cx = cx[1:]
    
    if flag > 0:
       tmp = cx>0.01*np.max(cx)
       t = np.arange(len(tmp))
       # Debug: print((tmp))
       # Find rising and falling edges
       t1 = t[1:][tmp[1:] > tmp[:-1]]
       t2 = t[:-1][tmp[:-1] > tmp[1:]]
       
       # Adjust t1 and t2 based on conditions
       if t1[0] > t2[0]:
           t2 = t2[1:]
       
       if t1[-1] > t2[-1]:
           t1 = t1[:-1]
       # Debug: print(len(t1))
       # Debug: print(len(t2))
       if len(t1) == len(t2) + 1:
           t1 = t1[:-1]
       
       if len(t2) == len(t1) + 1:
           t2 = t2[1:]
       
       # Initialize tmp and bla as empty lists
       tmp_list = []
       bla = []
       
       # Process intervals between t1 and t2
       for j in range(len(t1)):
           interval_sum = np.sum(cx[t1[j]:t2[j]])
           weighted_sum = cx[t1[j]:t2[j]] * tau[t1[j]:t2[j]] / interval_sum
           
           tmp_list.extend(weighted_sum)
           bla.append(interval_sum)
       
       # Normalize cx and tau
       cx = np.array(bla) / np.array(tmp_list)
       cx = cx / np.sum(cx)
       tau = np.array(tmp_list)
    
    return cx, tau, offset, csh, z, t, err 
            
            

def FluoFit(irf, y, p, dt, tau = None, lim = None,  flag_ml =  False, plt_flag = 1):
    """
    High-level multi-exponential fluorescence lifetime fitting with automatic initialization.
    
    This is the main user interface for fitting fluorescence decay data with multiple
    exponential components. It provides automatic lifetime estimation, parameter bounds
    handling, and flexible optimization options. The function can operate in both
    maximum likelihood and least squares modes.
    
    The fitted model is:
    y(t) = A₀ + Σᵢ Aᵢ * Convol(IRF_shifted(t), exp(-t/τᵢ) / (1-exp(-p/τᵢ)))
    
    Key Features:
    - Automatic lifetime initialization using DistFluoFit
    - Color shift optimization for IRF-decay alignment
    - Flexible parameter bounds specification
    - Choice between ML and LS optimization
    - Real-time fitting visualization
    - Comprehensive error analysis
    
    Parameters
    ----------
    irf : array-like, shape (n_time_bins,)
        Instrumental Response Function measurement.
        Should be normalized and background-subtracted.
    y : array-like, shape (n_time_bins,)
        Fluorescence decay data (photon counts per time bin).
        Must have same length as irf.
    p : float
        Time between laser excitation pulses (nanoseconds).
        Determines periodic boundary conditions and normalization.
    dt : float
        Time width of one TCSPC channel (nanoseconds).
        Converts time bins to absolute time units.
    tau : array-like, optional
        Initial guess for decay time constants (nanoseconds).
        If None, uses DistFluoFit for automatic estimation.
    lim : array-like, optional, shape (2*n_components,)
        Parameter bounds for decay times (nanoseconds).
        Format: [tau1_min, tau2_min, ..., tau1_max, tau2_max, ...]
        If None, uses default bounds [0.01, 100] ns for all components.
    flag_ml : bool, optional, default=False
        Optimization method selection:
        - True: Maximum Likelihood estimation
        - False: Least Squares estimation (with ML error metric)
    plt_flag : int, optional, default=1
        Plotting control:
        - None: No plots
        - Any other value: Display fitting results and residuals
        
    Returns
    -------
    tau1 : ndarray
        Fitted decay time constants (nanoseconds).
        Sorted from fastest to slowest components.
    A : ndarray
        Fitted amplitudes for each exponential component.
        Normalized such that sum(A) represents total amplitude.
    c : float
        Optimal color shift (nanoseconds).
        Positive values indicate IRF delayed relative to decay.
    z : ndarray
        Total fitted decay curve (sum of all components).
    zz : ndarray, shape (n_time_bins, n_components+1)
        Individual component curves:
        - Column 0: background offset
        - Columns 1+: individual exponential components
    offset : float
        Background level (constant offset).
    irs : ndarray
        IRF shifted by optimal color shift value.
    t : ndarray
        Time axis (nanoseconds) corresponding to the fitted curves.
    chi : float
        Goodness-of-fit metric (reduced chi-squared).
        
    Notes
    -----
    - Uses PIRLS (Poisson Iterative Reweighted Least Squares) for final amplitude fitting
    - Automatic parameter initialization via distributed lifetime analysis
    - Handles periodic boundary conditions for pulsed excitation
    - Color shift optimization accounts for timing misalignment
    - All time units converted consistently to nanoseconds
    
    Algorithm Overview
    ------------------
    1. Initialize lifetimes using DistFluoFit if not provided
    2. Convert all parameters to internal time units (dt)
    3. Set up optimization bounds and constraints
    4. Perform non-linear optimization (scipy.optimize.minimize)
    5. Extract optimal parameters and compute final fit
    6. Generate diagnostic plots if requested
    7. Convert results back to nanoseconds
    
    Examples
    --------
    >>> # Basic two-component fit with automatic initialization
    >>> tau_fit, A_fit, shift, fit_curve, components, bg, irf_shifted, t_axis, chi2 = FluoFit(
    ...     irf_data, decay_data, p=12.5, dt=0.032)
    
    >>> # Three-component fit with custom initialization and bounds
    >>> tau_init = [0.1, 1.0, 5.0]  # nanoseconds
    >>> bounds = [0.05, 0.5, 2.0, 0.2, 2.0, 10.0]  # [min1, min2, min3, max1, max2, max3]
    >>> results = FluoFit(irf_data, decay_data, p=12.5, dt=0.032,
    ...                   tau=tau_init, lim=bounds, flag_ml=True)
    
    >>> # Fitting with visualization disabled
    >>> results = FluoFit(irf_data, decay_data, p=12.5, dt=0.032, plt_flag=None)
    
    See Also
    --------
    DistFluoFit : Distributed lifetime analysis for initialization
    TCSPC_Fun : Low-level model function
    MLFit, LSFit : Optimization objective functions
    PIRLSnonneg : Weighted non-negative least squares solver
    """

    irf = np.array(irf).flatten()
    offset = 0;
    y = np.array(y).flatten()
    n = len(irf); 
    c = 0 # this will change if colorshift correction is necessary
    
    if tau is None: # get lifetimes guess values from DistFluoFit
        cx, tau, offset, c,_, _, _  = DistFluoFit(y, p, dt, irf)
        cx = np.array(cx).flatten() 

        # Identify where cx > 0
        tmp = cx > 0
        t = np.arange(len(tmp))  # Zero-indexed
        
        # Find indices where changes occur between positive and non-positive values
        t1 = t[np.where(tmp[1:] > tmp[:-1])[0] + 1]  # No adjustment needed, already zero-indexed
        t2 = t[np.where(tmp[:-1] > tmp[1:])[0]]  # No adjustment needed
        # Adjust t1 and t2 lengths if necessary
        if len(t1) == len(t2) + 1:
            t1 = t1[:-1]
            
        if len(t2) == len(t1) + 1:
            t2 = t2[1:]
        
        if t1[0] > t2[0]:
            t1 = t1[:-1]
            t2 = t2[1:]
                    
        # Initialize an empty list for the new tau values
        tmp_tau = []
        
        # Calculate the weighted average of tau values
        for j in range(len(t1)):
            cx_segment = cx[t1[j]:t2[j]+1]  # No need for -1/+1 adjustments anymore
            tau_segment = tau[t1[j]:t2[j]+1]
            weighted_tau = np.sum(cx_segment * tau_segment) / np.sum(cx_segment)
            tmp_tau.append(weighted_tau)
            
            # Update tau with the calculated values
        tau = np.array(tmp_tau)
        offset = 0
    else:
        c=0
        
    m = len(tau)
    
    
    if lim is None:
        lim_min =  np.array([0.01] * m)
        lim_max =  np.array([100.0] * m)
    else:
        lim_min = lim[:m]
        lim_max = lim[m:]
    
    lim_min /= dt
    lim_max /= dt    
    p /= dt
    tp = np.arange(1,p+1) # time axis
    t = np.arange(1,n+1)
    tau /= dt
    
    param = np.concatenate(([c], tau)) 
    # ecay times and Offset are assumed to be positive.
    paramin = np.concatenate(([-2/dt], lim_min))
    paramax =np.concatenate(([2/dt], lim_max))
    
    if flag_ml is True:
        res = minimize_s(lambda x: MLFit(x, y.astype(np.float64), irf, np.floor(p + 0.5)), param, bounds=list(zip(paramin, paramax)))
    else:    
        res = minimize_s(lambda x: LSFit(x, y.astype(np.float64), irf, np.floor(p + 0.5)), param, bounds=list(zip(paramin, paramax)))
    
    
    xfit = res.x
    c = xfit[0]
    tau = xfit[1:]
    x = np.exp(-(tp[:, None] - 1) * (1.0 / tau)) @ np.diag(1.0 / (1 - np.exp(-p / tau)))
    irs = (1 - c + np.floor(c)) * irf[(t - np.int_(c) - 1) % n] + \
         (c - np.floor(c)) * irf[(t - int(np.ceil(c)) - 1) % n]
         
    z = Convol(irs, x);
    z = np.column_stack((np.ones(len(z)), z))
    # Linear least squares solution for A
    # A, _, _, _ = lstsq(z, y) 
    TINY = 10**-10
    # A,_ = nnls(z, y)
    A,_ = PIRLSnonneg(z,y,10)
    zz = z*A;
    z = z @ A      
    if plt_flag is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        ax1.semilogy(t, y, 'bo', label="y")
        ax1.semilogy(t, z, 'r',label="fitted z")
        ax1.semilogy(t,irf/np.max(irf)*np.max(y),'k' ,label = 'irf')
        ax1.legend()
        ax1.set_ylim(bottom=np.min(y)) 
        ax1.set_ylim(top=np.max(y)*2) 
        
        ax2.plot(t,(y-z)/np.sqrt(np.abs(z)))
        ax2.axhline(0, color='black', lw=1, linestyle='--')
        
        plt.draw()
        plt.pause(0.001)
    chi = np.sum((y-z-TINY)**2/ np.abs(z+TINY))/(n-m);
    t = dt*t;
    tau1 = dt*tau
    c = dt*c
    offset = zz[0]
    A = A[1:]    
    
    return tau1, A, c, z, zz, offset, irs, t, chi
    # 
    # 
    
        
    
def PatternMatchIm(y,M,mode = 'Default'):
     """
     y : array, shape (nx, ny, t)
     M : array, shape (t, n_basis)
     mode : 'Default' | 'Nonneg' | 'PIRLS'
     
     Returns
     -------
     C : array, shape (nx, ny, n_basis)
          coefficient maps
          Z : array, shape (nx, ny, t)
          reconstructed data
          """
     nx, ny, t = y.shape
     t2, n_basis = M.shape
     M = M/np.sum(M,axis=0) # normalize the patterns 
     assert t2 == t, "time‐axis of y and M must match."
     
     # reshape y into (t, nx*ny)
     Y = y.reshape(nx*ny, t).T     # now shape = (t, npix)
     
     if mode == 'Default':
         # solve M @ C = Y  in one go:
         # np.linalg.lstsq treats columns of Y as separate RHS
         C_flat, *_ = np.linalg.lstsq(M, Y, rcond=None)
         # C_flat is (n_basis, npix)
         
     elif mode == 'Nonneg':
         # sklearn can do multi‐target nonneg‐LS in one shot:
         #   fit X=M, Y=Y.T (shape: samples×features, samples×targets)
         lr = LinearRegression(positive=True, fit_intercept=False)
         lr.fit(M, Y)
         # lr.coef_ is (n_targets, n_features) = (npix, n_basis)
         C_flat = lr.coef_.T          # make (n_basis, npix)
         
     elif mode == 'PIRLS':
         # Use GPU-accelerated PIRLS with automatic fallback to CPU
         C_flat = PIRLSnonneg_batch_gpu(M, Y)
         
         #C_flat = PIRLSnonneg_batch(M, Y)
   
      
     else:
        raise ValueError("Unknown mode")

     # reconstruct
     Z_flat = M @ C_flat            # shape (t, npix)

     # reshape back to (nx,ny,...)
     C = C_flat.T.reshape(nx, ny, n_basis)
     Z = Z_flat.T.reshape(nx, ny, t)
     return C, Z
