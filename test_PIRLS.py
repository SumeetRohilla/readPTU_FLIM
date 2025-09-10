# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:24:51 2025

@author: narai
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import time
#sys.path.append(str(Path(file).parent.parent))

from FLIM_fitter import IRF_Fun, Convol, PIRLSnonneg
from scipy.optimize import nnls, lsq_linear

# Try to import PyCUDA PIRLS
try:
    from PIRLS_cu.pirls_pycuda import pirls_batch_cuda, CUDA_AVAILABLE
    print("✅ PyCUDA PIRLS available")
except ImportError as e:
    print(f"⚠️  PyCUDA PIRLS not available: {e}")
    CUDA_AVAILABLE = False


def create_simple_test_data():
    # Time axis (256 bins, 32 ps resolution)
    n_bins = 708
    dt = 0.032  # ns
    time_axis = np.arange(n_bins) * dt
    
    # Simple IRF (Gaussian)
    irf_params = [0.5, 0.03, 0.05, 0.1, 0.005, 0.05, 0.0]
    irf = IRF_Fun(irf_params, time_axis)
    
    # Two exponential components
    lifetimes = [0.5, 2.0, 6.0]  # ns
    true_amplitudes = [0.0001, 0.3, 0.5-0.0001, 0.2]  # [background, fast, slow]
    
    # Generate components matrix
    components = np.zeros((n_bins, len(true_amplitudes)))
    components[:, 0] = 1.0  # Background
    
    for i, lifetime in enumerate(lifetimes):
        t_rel = time_axis - irf_params[0]
        decay = np.exp(-t_rel / lifetime) / lifetime
        decay[t_rel < 0] = 0
        convolved = Convol(irf, decay)
        components[:, i + 1] = convolved / np.sum(convolved)
        
    # Generate synthetic data
    ideal_curve = components @ np.array(true_amplitudes)
    ideal_curve = ideal_curve / np.sum(ideal_curve) * 1000  # 5k counts
    
    # Add Poisson noise
    noisy_data = np.random.poisson(ideal_curve)

    return components, noisy_data, time_axis, true_amplitudes


def test_all_methods():
    print("Simple TCSPC Fitting Validation Test")
    print("=" * 40)
    
    # Generate test data
    components, data, taxis, true_amps = create_simple_test_data()
    true_amps = np.array(true_amps)
    
    print(f"Test data: {len(data)} time bins, {np.sum(data)} total counts")
    print(f"True amplitudes: {true_amps}")
    print()
    
    # Test each method
    methods = {
        'scipy_nnls': lambda: nnls(components, data)[0],
        'scipy_bounded_ls': lambda: lsq_linear(components, data, bounds=(0, np.inf)).x,
        'pirls': lambda: PIRLSnonneg(components, data, max_num_iter=10)[0]
    }
    
    # Add PyCUDA method if available
    if CUDA_AVAILABLE:
        def cuda_pirls_single():
            # PyCUDA expects (n_samples, n_pixels) for Y, but we have single curve
            Y_batch = data.reshape(-1, 1).astype(np.float32)
            M_f32 = components.astype(np.float32)
            Beta = pirls_batch_cuda(M_f32, Y_batch, max_iter=10)
            print(Beta)
            return Beta.flatten()  # Return as 1D array
        
        methods['pirls_cuda'] = cuda_pirls_single

    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for method_name, method_func in methods.items():
        try:
            # Time the method
            start_time = time.time()
            fitted_amps = method_func()
            elapsed_time = time.time() - start_time
            
            Zfit = components @ fitted_amps
            
            # Plot data and fit
            if method_name == 'scipy_nnls':
                plt.semilogy(taxis, data, 'ko', markersize=2, label='Data', alpha=0.7)
            plt.semilogy(taxis, Zfit, linewidth=2, label=f'{method_name} fit')
            
            # Normalize amplitudes for comparison
            fitted_amps_norm = fitted_amps / np.sum(fitted_amps)
            rmse = np.sqrt(np.mean((fitted_amps_norm - true_amps) ** 2))
            
            results[method_name] = {
                'amplitudes': fitted_amps,
                'amplitudes_norm': fitted_amps_norm,
                'rmse': rmse,
                'time': elapsed_time,
                'success': True
            }
            
            print(f"{method_name:15s}: RMSE = {rmse:.4f}, Time = {elapsed_time:.4f}s")
            print(f"                 Fitted: {fitted_amps_norm}")
            print()
            
        except Exception as e:
            results[method_name] = {
                'error': str(e),
                'success': False
            }
            print(f"{method_name:15s}: FAILED - {e}")
            print()
    
    # Finalize plot
    plt.xlabel('Time (ns)')
    plt.ylabel('Counts')
    plt.title('TCSPC Fitting Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
            
    # Summary
    successful_methods = [name for name, result in results.items() if result['success']]
    print(f"Successfully tested {len(successful_methods)}/{len(methods)} methods")
    
    if successful_methods:
        best_method = min(successful_methods,
                          key=lambda x: results[x]['rmse'])
        fastest_method = min(successful_methods,
                           key=lambda x: results[x]['time'])
        
        print(f"Most accurate: {best_method} (RMSE: {results[best_method]['rmse']:.4f})")
        print(f"Fastest method: {fastest_method} (Time: {results[fastest_method]['time']:.4f}s)")
        
        # Show speedup if CUDA was tested
        if CUDA_AVAILABLE and 'pirls_cuda' in results and results['pirls_cuda']['success']:
            cpu_time = results['pirls']['time'] if 'pirls' in results and results['pirls']['success'] else 0
            gpu_time = results['pirls_cuda']['time']
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"CUDA speedup: {speedup:.2f}x faster than CPU PIRLS")
        
    return results



results = test_all_methods()

