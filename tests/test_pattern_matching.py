#!/usr/bin/env python3
"""
Comprehensive test suite for PatternMatchIm function in FLIM_fitter.py

This module tests the PatternMatchIm function with all three modes:
- 'Default': Standard least squares
- 'Nonneg': Non-negative least squares 
- 'PIRLS': GPU-accelerated Poisson Iterative Reweighted Least Squares

Tests use synthetic TCSPC data generated with IRF_Fun and various decay models.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Add parent directory to path to import FLIM_fitter
sys.path.append(str(Path(__file__).parent.parent))

try:
    from FLIM_fitter import IRF_Fun, Convol, PatternMatchIm, PIRLSnonneg
    print("✓ Successfully imported FLIM_fitter functions")
    FLIM_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import FLIM_fitter: {e}")
    print("Make sure FLIM_fitter.py is in the parent directory")
    FLIM_AVAILABLE = False

# Check for GPU dependencies
try:
    import cupy as cp
    import numba.cuda as cuda
    GPU_AVAILABLE = cuda.is_available()
    print(f"✓ GPU support: {'Available' if GPU_AVAILABLE else 'Not available'}")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ GPU support: Not available (cupy/numba-cuda not installed)")

class TestConfig:
    """Configuration parameters for tests"""
    # Image dimensions
    nx_small = 8
    ny_small = 8
    nx_large = 32
    ny_large = 32
    
    # Time axis parameters
    n_channels = 781  # 25 ns / 0.032 ns per channel = 781 channels
    dt = 0.032  # ns per channel
    
    # IRF parameters (realistic for typical TCSPC systems)
    irf_params = [
        2.0,    # t_0: peak position (ns)
        0.15,   # w1: Gaussian width
        0.05,   # T1: first time constant
        0.1,    # T2: second time constant
        0.01,   # a: reconvolution amplitude
        0.05,   # b: tail amplitude
        0.0     # dt: time delay
    ]
    
    # Decay models for testing
    decay_models = {
        'single_exp': {
            'lifetimes': [2.5],
            'amplitudes': [1.0],
            'description': 'Single exponential decay'
        },
        'bi_exp': {
            'lifetimes': [0.8, 3.5],
            'amplitudes': [0.3, 0.7],
            'description': 'Bi-exponential decay'
        },
        'tri_exp': {
            'lifetimes': [0.3, 1.8, 5.2],
            'amplitudes': [0.2, 0.5, 0.3],
            'description': 'Tri-exponential decay'
        }
    }
    
    # Noise parameters
    max_counts = 1000
    background_level = 10

def create_time_axis(n_channels=256, dt=0.032):
    """Create time axis for TCSPC data"""
    return np.arange(n_channels) * dt

def generate_synthetic_irf(time_axis, irf_params=None):
    """
    Generate synthetic IRF using IRF_Fun
    
    Parameters
    ----------
    time_axis : ndarray
        Time points (ns)
    irf_params : list, optional
        IRF parameters [t_0, w1, T1, T2, a, b, dt]
        
    Returns
    -------
    irf : ndarray
        Normalized IRF
    """
    if irf_params is None:
        irf_params = TestConfig.irf_params
    
    irf = IRF_Fun(irf_params, time_axis)
    return irf / np.sum(irf)  # Normalize

def generate_decay_component(time_axis, lifetime, amplitude=1.0):
    """
    Generate single exponential decay component
    
    Parameters
    ----------
    time_axis : ndarray
        Time points (ns)
    lifetime : float
        Decay lifetime (ns)
    amplitude : float
        Component amplitude
        
    Returns
    -------
    decay : ndarray
        Exponential decay function
    """
    return amplitude * np.exp(-time_axis / lifetime)

def create_synthetic_tcspc_data(nx, ny, n_channels, decay_model, noise=True, 
                               spatial_variation=True, dt=0.032):
    """
    Create synthetic TCSPC data with known ground truth
    
    Parameters
    ----------
    nx, ny : int
        Image dimensions
    n_channels : int
        Number of time channels
    decay_model : dict
        Contains 'lifetimes' and 'amplitudes' lists
    noise : bool
        Whether to add Poisson noise
    spatial_variation : bool
        Whether to add spatial variation in amplitudes
    dt : float
        Time resolution (ns)
        
    Returns
    -------
    data : ndarray, shape (nx, ny, n_channels)
        Synthetic TCSPC data
    ground_truth : dict
        Contains true parameters for validation
    """
    print(f"Creating synthetic data: {nx}×{ny}×{n_channels}")
    print(f"Decay model: {decay_model['description']}")
    
    # Create time axis
    time_axis = create_time_axis(n_channels, dt)
    
    # Generate IRF
    irf = generate_synthetic_irf(time_axis)
    
    # Create spatial amplitude maps
    lifetimes = decay_model['lifetimes']
    amplitudes = decay_model['amplitudes']
    n_components = len(lifetimes)
    
    # Initialize data array
    data = np.zeros((nx, ny, n_channels))
    amplitude_maps = np.zeros((nx, ny, n_components))
    
    # Create spatial variation patterns
    if spatial_variation:
        x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
        for i, amp in enumerate(amplitudes):
            # Create smooth spatial variation
            variation = 0.5 + 0.3 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
            amplitude_maps[:, :, i] = amp * variation
    else:
        for i, amp in enumerate(amplitudes):
            amplitude_maps[:, :, i] = amp
    
    # Generate decay data for each pixel
    for i in range(nx):
        for j in range(ny):
            pixel_decay = np.zeros(n_channels)
            
            # Add background
            pixel_decay += TestConfig.background_level
            
            # Add each decay component
            for k, (lifetime, amplitude) in enumerate(zip(lifetimes, amplitudes)):
                component = generate_decay_component(time_axis, lifetime, 
                                                   amplitude_maps[i, j, k])
                # Convolve with IRF
                convolved = Convol(irf, component)
                pixel_decay += TestConfig.max_counts * convolved[:n_channels]
            
            # Add Poisson noise
            if noise:
                pixel_decay = np.random.poisson(pixel_decay)
            
            data[i, j, :] = pixel_decay
    
    # Store ground truth
    ground_truth = {
        'lifetimes': lifetimes,
        'amplitudes': amplitudes,
        'amplitude_maps': amplitude_maps,
        'irf': irf,
        'time_axis': time_axis,
        'background': TestConfig.background_level
    }
    
    print(f"✓ Generated data with max counts: {np.max(data):.0f}")
    return data, ground_truth

def create_basis_matrix(time_axis, lifetimes, irf):
    """
    Create basis matrix for pattern matching
    
    Parameters
    ----------
    time_axis : ndarray
        Time points
    lifetimes : list
        Component lifetimes
    irf : ndarray
        Instrumental response function
        
    Returns
    -------
    M : ndarray, shape (n_channels, n_components + 1)
        Basis matrix with background column
    """
    n_channels = len(time_axis)
    n_components = len(lifetimes)
    
    # Initialize matrix with background column
    M = np.zeros((n_channels, n_components + 1))
    M[:, 0] = 1.0  # Background/offset column
    
    # Add convolved decay components
    for i, lifetime in enumerate(lifetimes):
        decay = generate_decay_component(time_axis, lifetime)
        convolved = Convol(irf, decay)
        M[:, i + 1] = convolved[:n_channels]
    
    return M

def test_pattern_matching_modes(data, M, mode_name):
    """
    Test PatternMatchIm with specified mode
    
    Parameters
    ----------
    data : ndarray
        TCSPC data
    M : ndarray
        Basis matrix
    mode_name : str
        Pattern matching mode
        
    Returns
    -------
    results : dict
        Test results including timing and accuracy
    """
    print(f"\n--- Testing mode: {mode_name} ---")
    
    # Skip PIRLS mode if GPU not available
    if mode_name == 'PIRLS' and not GPU_AVAILABLE:
        print("⚠ Skipping PIRLS mode (GPU not available)")
        return {'skipped': True, 'reason': 'GPU not available'}
    
    try:
        start_time = time.time()
        
        # Run pattern matching
        C, Z = PatternMatchIm(data, M, mode=mode_name)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate performance metrics
        nx, ny, n_channels = data.shape
        total_pixels = nx * ny
        pixels_per_second = total_pixels / elapsed_time
        
        # Calculate reconstruction error
        mse = np.mean((data - Z) ** 2)
        rmse = np.sqrt(mse)
        relative_error = rmse / np.mean(data)
        
        results = {
            'success': True,
            'elapsed_time': elapsed_time,
            'pixels_per_second': pixels_per_second,
            'mse': mse,
            'rmse': rmse,
            'relative_error': relative_error,
            'coefficients': C,
            'reconstruction': Z,
            'mode': mode_name
        }
        
        print(f"✓ {mode_name} completed successfully")
        print(f"  Time: {elapsed_time:.3f} s")
        print(f"  Performance: {pixels_per_second:.1f} pixels/s")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Relative error: {relative_error:.1%}")
        
        return results
        
    except Exception as e:
        print(f"✗ {mode_name} failed: {e}")
        return {'success': False, 'error': str(e), 'mode': mode_name}

def validate_results(results, ground_truth, tolerance=0.2):
    """
    Validate pattern matching results against ground truth
    
    Parameters
    ----------
    results : dict
        Pattern matching results
    ground_truth : dict
        Known true parameters
    tolerance : float
        Acceptable relative error
        
    Returns
    -------
    validation : dict
        Validation results
    """
    if not results.get('success', False):
        return {'valid': False, 'reason': 'Pattern matching failed'}
    
    # Check reconstruction quality
    relative_error = results['relative_error']
    reconstruction_valid = relative_error < tolerance
    
    # Check coefficient extraction (compare amplitudes)
    C = results['coefficients']
    true_amplitudes = ground_truth['amplitude_maps']
    
    # Extract non-background coefficients
    extracted_amplitudes = C[:, :, 1:]  # Skip background column
    
    # Calculate amplitude correlation
    amplitude_correlations = []
    for k in range(true_amplitudes.shape[2]):
        true_flat = true_amplitudes[:, :, k].flatten()
        extracted_flat = extracted_amplitudes[:, :, k].flatten()
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(true_flat, extracted_flat)[0, 1]
        amplitude_correlations.append(correlation)
    
    mean_correlation = np.mean(amplitude_correlations)
    coefficients_valid = mean_correlation > 0.8  # 80% correlation threshold
    
    validation = {
        'valid': reconstruction_valid and coefficients_valid,
        'reconstruction_valid': reconstruction_valid,
        'coefficients_valid': coefficients_valid,
        'relative_error': relative_error,
        'mean_correlation': mean_correlation,
        'amplitude_correlations': amplitude_correlations
    }
    
    return validation

def plot_test_results(data, results_dict, ground_truth, save_path=None):
    """
    Create comprehensive plots of test results
    
    Parameters
    ----------
    data : ndarray
        Original TCSPC data
    results_dict : dict
        Results from all tested modes
    ground_truth : dict
        Ground truth parameters
    save_path : str, optional
        Path to save plots
    """
    successful_results = {k: v for k, v in results_dict.items() 
                         if v.get('success', False)}
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    n_modes = len(successful_results)
    fig, axes = plt.subplots(3, n_modes, figsize=(4*n_modes, 12))
    
    if n_modes == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot for each successful mode
    for col, (mode_name, results) in enumerate(successful_results.items()):
        Z = results['reconstruction']
        C = results['coefficients']
        
        # 1. Sample decay curves (center pixel)
        center_x, center_y = data.shape[0]//2, data.shape[1]//2
        time_axis = ground_truth['time_axis']
        
        axes[0, col].semilogy(time_axis, data[center_x, center_y, :], 'b.', 
                             label='Original', alpha=0.7)
        axes[0, col].semilogy(time_axis, Z[center_x, center_y, :], 'r-', 
                             label='Fitted', linewidth=2)
        axes[0, col].set_xlabel('Time (ns)')
        axes[0, col].set_ylabel('Counts')
        axes[0, col].set_title(f'{mode_name} - Decay Fit')
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)
        
        # 2. Amplitude maps (first component)
        if C.shape[2] > 1:  # More than just background
            amp_map = C[:, :, 1]  # First non-background component
            im1 = axes[1, col].imshow(amp_map, cmap='viridis')
            axes[1, col].set_title(f'{mode_name} - Component 1 Amplitude')
            plt.colorbar(im1, ax=axes[1, col])
        
        # 3. Residuals map
        residuals = np.sum((data - Z)**2, axis=2)
        im2 = axes[2, col].imshow(residuals, cmap='hot')
        axes[2, col].set_title(f'{mode_name} - Residuals')
        plt.colorbar(im2, ax=axes[2, col])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plots saved to {save_path}")
    
    plt.show()

def run_comprehensive_test(test_size='small', decay_model_name='bi_exp'):
    """
    Run comprehensive test of PatternMatchIm function
    
    Parameters
    ----------
    test_size : str
        'small' or 'large' - determines image size
    decay_model_name : str
        Key from TestConfig.decay_models
        
    Returns
    -------
    test_summary : dict
        Complete test results summary
    """
    print(f"\n{'='*60}")
    print(f"RUNNING COMPREHENSIVE PATTERN MATCHING TEST")
    print(f"Test size: {test_size}")
    print(f"Decay model: {decay_model_name}")
    print(f"{'='*60}")
    
    # Set up test parameters
    if test_size == 'small':
        nx, ny = TestConfig.nx_small, TestConfig.ny_small
    else:
        nx, ny = TestConfig.nx_large, TestConfig.ny_large
    
    decay_model = TestConfig.decay_models[decay_model_name]
    
    # Generate synthetic data
    print("\n1. Generating synthetic TCSPC data...")
    data, ground_truth = create_synthetic_tcspc_data(
        nx, ny, TestConfig.n_channels, decay_model, 
        noise=True, spatial_variation=True, dt=TestConfig.dt
    )
    
    # Create basis matrix
    print("\n2. Creating basis matrix...")
    M = create_basis_matrix(
        ground_truth['time_axis'], 
        ground_truth['lifetimes'], 
        ground_truth['irf']
    )
    print(f"✓ Basis matrix shape: {M.shape}")
    
    # Test all modes
    print("\n3. Testing pattern matching modes...")
    modes_to_test = ['Default', 'Nonneg', 'PIRLS']
    results_dict = {}
    
    for mode in modes_to_test:
        results_dict[mode] = test_pattern_matching_modes(data, M, mode)
    
    # Validate results
    print("\n4. Validating results...")
    validation_dict = {}
    for mode, results in results_dict.items():
        if results.get('success', False):
            validation_dict[mode] = validate_results(results, ground_truth)
            valid = validation_dict[mode]['valid']
            corr = validation_dict[mode]['mean_correlation']
            print(f"  {mode}: {'✓ Valid' if valid else '✗ Invalid'} "
                  f"(correlation: {corr:.3f})")
    
    # Create plots
    print("\n5. Generating plots...")
    plot_filename = f"pattern_matching_{test_size}_{decay_model_name}_results.png"
    plot_test_results(data, results_dict, ground_truth, plot_filename)
    
    # Summary
    test_summary = {
        'test_size': test_size,
        'decay_model': decay_model_name,
        'data_shape': data.shape,
        'results': results_dict,
        'validation': validation_dict,
        'ground_truth': ground_truth
    }
    
    return test_summary

def print_test_summary(test_summary):
    """Print formatted test summary"""
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    print(f"Test configuration:")
    print(f"  Size: {test_summary['test_size']}")
    print(f"  Data shape: {test_summary['data_shape']}")
    print(f"  Decay model: {test_summary['decay_model']}")
    
    print(f"\nResults by mode:")
    for mode, results in test_summary['results'].items():
        if results.get('skipped', False):
            print(f"  {mode}: SKIPPED ({results['reason']})")
        elif results.get('success', False):
            validation = test_summary['validation'].get(mode, {})
            valid_status = "✓ Valid" if validation.get('valid', False) else "✗ Invalid"
            print(f"  {mode}: SUCCESS - {valid_status}")
            print(f"    Performance: {results['pixels_per_second']:.1f} pixels/s")
            print(f"    Accuracy: {validation.get('mean_correlation', 0):.3f} correlation")
        else:
            print(f"  {mode}: FAILED ({results.get('error', 'Unknown error')})")

def run_all_pattern_tests():
    """
    Run all pattern matching tests - Spyder-friendly version
    Call this function directly in Spyder console
    """
    if not FLIM_AVAILABLE:
        print("✗ Cannot run tests: FLIM_fitter not available")
        return []
    
    print("Starting PatternMatchIm comprehensive test suite...")
    
    # Test different configurations
    test_configs = [
        ('small', 'single_exp'),
        ('small', 'bi_exp'),
        ('small', 'tri_exp'),
    ]
    
    # Add large tests if GPU is available
    if GPU_AVAILABLE:
        test_configs.append(('large', 'tri_exp'))
    
    all_results = []
    
    for test_size, decay_model in test_configs:
        try:
            summary = run_comprehensive_test(test_size, decay_model)
            all_results.append(summary)
            print_test_summary(summary)
        except Exception as e:
            print(f"✗ Test failed: {test_size}, {decay_model} - {e}")
    
    print(f"\n{'='*60}")
    print(f"ALL TESTS COMPLETED")
    print(f"Total test configurations: {len(test_configs)}")
    print(f"Successful tests: {len(all_results)}")
    print(f"{'='*60}")
    
    return all_results

def create_amplitude_test_data(nx=16, ny=16, n_channels=781, dt=0.032):
    """
    Create 2D TCSPC data with spatially varying tri-exponential decay amplitudes
    
    Parameters
    ----------
    nx, ny : int
        Image dimensions
    n_channels : int
        Number of time channels
    dt : float
        Time resolution (ns)
        
    Returns
    -------
    data : ndarray, shape (nx, ny, n_channels)
        Synthetic TCSPC data with varying amplitudes
    ground_truth : dict
        Contains true parameters and amplitude patterns
    """
    print(f"Creating amplitude test data: {nx}×{ny}×{n_channels}")
    
    # Use tri-exponential decay model
    decay_model = TestConfig.decay_models['tri_exp']
    lifetimes = decay_model['lifetimes']  # [0.3, 1.8, 5.2] ns
    base_amplitudes = decay_model['amplitudes']  # [0.2, 0.5, 0.3]
    
    # Create time axis and IRF
    time_axis = create_time_axis(n_channels, dt)
    irf = generate_synthetic_irf(time_axis)
    
    # Create distinct spatial patterns for each component
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
    
    # Component 1 (fast): Radial gradient from center
    center_x, center_y = nx//2, ny//2
    r = np.sqrt((np.arange(nx)[:, None] - center_x)**2 + (np.arange(ny) - center_y)**2)
    r_norm = r / np.max(r)
    pattern1 = 0.1 + 0.4 * (1 - r_norm)  # Higher in center
    
    # Component 2 (medium): Horizontal stripes
    pattern2 = 0.2 + 0.6 * (0.5 + 0.5 * np.sin(4 * np.pi * x))
    
    # Component 3 (slow): Checkerboard pattern
    check_x = np.floor(x * 4) % 2
    check_y = np.floor(y * 4) % 2
    pattern3 = 0.1 + 0.5 * (check_x == check_y)
    
    # Normalize patterns to maintain realistic ratios
    total_pattern = pattern1 + pattern2 + pattern3
    pattern1 = pattern1 / total_pattern * base_amplitudes[0] * 3
    pattern2 = pattern2 / total_pattern * base_amplitudes[1] * 3
    pattern3 = pattern3 / total_pattern * base_amplitudes[2] * 3
    
    amplitude_maps = np.stack([pattern1, pattern2, pattern3], axis=2)
    
    # Generate TCSPC data
    data = np.zeros((nx, ny, n_channels))
    
    for i in range(nx):
        for j in range(ny):
            pixel_decay = np.zeros(n_channels)
            
            # Add background
            pixel_decay += TestConfig.background_level
            
            # Add each decay component with spatial amplitude
            for k, lifetime in enumerate(lifetimes):
                component = generate_decay_component(time_axis, lifetime, amplitude_maps[i, j, k])
                convolved = Convol(irf, component)
                pixel_decay += TestConfig.max_counts * convolved[:n_channels]
            
            # Add Poisson noise
            pixel_decay = np.random.poisson(pixel_decay)
            data[i, j, :] = pixel_decay
    
    # Store ground truth
    ground_truth = {
        'lifetimes': lifetimes,
        'base_amplitudes': base_amplitudes,
        'amplitude_maps': amplitude_maps,
        'amplitude_patterns': {
            'component_1_fast': pattern1,
            'component_2_medium': pattern2,
            'component_3_slow': pattern3
        },
        'irf': irf,
        'time_axis': time_axis,
        'background': TestConfig.background_level,
        'description': 'Tri-exponential with distinct spatial amplitude patterns'
    }
    
    print(f"✓ Generated amplitude test data")
    print(f"  Component 1 (fast, {lifetimes[0]} ns): Radial pattern")
    print(f"  Component 2 (medium, {lifetimes[1]} ns): Horizontal stripes")
    print(f"  Component 3 (slow, {lifetimes[2]} ns): Checkerboard pattern")
    print(f"  Max counts: {np.max(data):.0f}")
    
    return data, ground_truth

def plot_amplitude_recovery_results(data, results_dict, ground_truth, save_path=None):
    """
    Create comprehensive amplitude recovery plots
    
    Parameters
    ----------
    data : ndarray
        Original TCSPC data
    results_dict : dict
        Results from all tested modes
    ground_truth : dict
        Ground truth parameters including amplitude patterns
    save_path : str, optional
        Path to save plots
    """
    successful_results = {k: v for k, v in results_dict.items()
                         if v.get('success', False)}
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    n_modes = len(successful_results)
    n_components = len(ground_truth['lifetimes'])
    
    # Create figure with subplots: 4 rows (ground truth + 3 components) × n_modes columns
    fig, axes = plt.subplots(4, n_modes, figsize=(5*n_modes, 16))
    
    if n_modes == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot for each successful mode
    for col, (mode_name, results) in enumerate(successful_results.items()):
        C = results['coefficients']
        C = C/np.sum(C[:,:,1:],axis=2, keepdims=True)
        
        # Row 0: Sample decay curve from center pixel
        center_x, center_y = data.shape[0]//2, data.shape[1]//2
        time_axis = ground_truth['time_axis']
        
        axes[0, col].semilogy(time_axis, data[center_x, center_y, :], 'b.',
                             label='Original', alpha=0.7, markersize=3)
        axes[0, col].semilogy(time_axis, results['reconstruction'][center_x, center_y, :], 'r-',
                             label='Fitted', linewidth=2)
        axes[0, col].set_xlabel('Time (ns)')
        axes[0, col].set_ylabel('Counts')
        axes[0, col].set_title(f'{mode_name} - Center Pixel Fit')
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)
        
        # Rows 1-3: Amplitude maps for each component
        for comp in range(n_components):
            row = comp + 1
            
            # Extract amplitude map (skip background column 0)
            if C.shape[2] > comp + 1:  # Check if component exists
                recovered_amp = C[:, :, comp + 1]  # Component comp (1-indexed due to background)
                true_amp = ground_truth['amplitude_maps'][:, :, comp]
                
                # Plot recovered amplitude
                im = axes[row, col].imshow(recovered_amp, cmap='viridis',
                                         vmin=0, vmax=np.max(true_amp)*1.1)
                axes[row, col].set_title(f'{mode_name} - Component {comp+1}\n'
                                       f'({ground_truth["lifetimes"][comp]:.1f} ns)')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
                
                # Calculate and display correlation
                correlation = np.corrcoef(true_amp.flatten(), recovered_amp.flatten())[0, 1]
                axes[row, col].text(0.02, 0.98, f'r = {correlation:.3f}',
                                  transform=axes[row, col].transAxes,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                  verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Amplitude recovery plots saved to {save_path}")
    
    plt.show()
    
    # Create a separate figure showing ground truth patterns
    fig2, axes2 = plt.subplots(1, n_components, figsize=(4*n_components, 4))
    if n_components == 1:
        axes2 = [axes2]
    
    for comp in range(n_components):
        true_amp = ground_truth['amplitude_maps'][:, :, comp]
        im = axes2[comp].imshow(true_amp, cmap='viridis')
        axes2[comp].set_title(f'Ground Truth - Component {comp+1}\n'
                            f'({ground_truth["lifetimes"][comp]:.1f} ns)')
        plt.colorbar(im, ax=axes2[comp], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        gt_save_path = save_path.replace('.png', '_ground_truth.png')
        plt.savefig(gt_save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ground truth plots saved to {gt_save_path}")
    
    plt.show()

def run_amplitude_recovery_test():
    """
    Run amplitude recovery test with tri-exponential decay and spatial patterns
    
    This test creates 2D TCSPC data with distinct spatial amplitude patterns
    for each decay component and evaluates how well each mode recovers them.
    """
    print(f"\n{'='*60}")
    print(f"RUNNING AMPLITUDE RECOVERY TEST")
    print(f"{'='*60}")
    
    # Generate test data with spatial amplitude patterns (25 ns time window)
    print("\n1. Generating spatially-varying tri-exponential data...")
    n_channels_25ns = int(25.0 / 0.032)  # 25 ns / 0.032 ns per channel = 781 channels
    data, ground_truth = create_amplitude_test_data(nx=16, ny=16, n_channels=n_channels_25ns)
    
    # Create basis matrix
    print("\n2. Creating basis matrix...")
    M = create_basis_matrix(
        ground_truth['time_axis'],
        ground_truth['lifetimes'],
        ground_truth['irf']
    )
    print(f"✓ Basis matrix shape: {M.shape}")
    
    # Test all modes
    print("\n3. Testing pattern matching modes...")
    modes_to_test = ['Default', 'Nonneg', 'PIRLS']
    results_dict = {}
    
    for mode in modes_to_test:
        results_dict[mode] = test_pattern_matching_modes(data, M, mode)
    
    # Validate amplitude recovery
    print("\n4. Validating amplitude recovery...")
    for mode, results in results_dict.items():
        if results.get('success', False):
            C = results['coefficients']
            
            # Calculate correlation for each component
            print(f"\n  {mode} amplitude recovery:")
            for comp in range(len(ground_truth['lifetimes'])):
                if C.shape[2] > comp + 1:
                    recovered = C[:, :, comp + 1].flatten()
                    true = ground_truth['amplitude_maps'][:, :, comp].flatten()
                    correlation = np.corrcoef(true, recovered)[0, 1]
                    rmse = np.sqrt(np.mean((recovered - true)**2))
                    print(f"    Component {comp+1} ({ground_truth['lifetimes'][comp]:.1f} ns): "
                          f"r = {correlation:.3f}, RMSE = {rmse:.3f}")
    
    # Create amplitude recovery plots
    print("\n5. Generating amplitude recovery plots...")
    plot_filename = "amplitude_recovery_test_results.png"
    plot_amplitude_recovery_results(data, results_dict, ground_truth, plot_filename)
    
    # Summary
    test_summary = {
        'test_type': 'amplitude_recovery',
        'data_shape': data.shape,
        'results': results_dict,
        'ground_truth': ground_truth
    }
    
    print(f"\n{'='*60}")
    print("AMPLITUDE RECOVERY TEST SUMMARY")
    print(f"{'='*60}")
    
    for mode, results in results_dict.items():
        if results.get('success', False):
            print(f"\n{mode}:")
            print(f"  Time: {results['elapsed_time']:.3f} s")
            print(f"  Performance: {results['pixels_per_second']:.1f} pixels/s")
            print(f"  Overall RMSE: {results['rmse']:.4f}")
    
    return test_summary

def quick_demo():
    """Quick demonstration of PatternMatchIm - ideal for Spyder"""
    if not FLIM_AVAILABLE:
        print("✗ Cannot run demo: FLIM_fitter not available")
        return None
        
    print("Running quick PatternMatchIm demo...")
    return run_comprehensive_test('small', 'bi_exp')

def amplitude_demo():
    """Quick amplitude recovery demonstration - ideal for Spyder"""
    if not FLIM_AVAILABLE:
        print("✗ Cannot run demo: FLIM_fitter not available")
        return None
        
    print("Running amplitude recovery demo...")
    return run_amplitude_recovery_test()

if __name__ == "__main__":
    # For Spyder compatibility - don't use sys.exit()
    run_all_pattern_tests()