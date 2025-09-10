#!/usr/bin/env python3
"""
Test script for PyCUDA PIRLS implementation
"""

import numpy as np
import sys
import time
from pathlib import Path

try:
    from pirls_pycuda import pirls_batch_cuda, CUDA_AVAILABLE
    print("Successfully imported CUDA PIRLS module")
except ImportError as e:
    print(f"Failed to import CUDA PIRLS: {e}")
    print("Make sure PyCUDA is installed: pip install pycuda")
    sys.exit(1)

def compare_with_scipy():
    """Compare CUDA PIRLS with scipy NNLS for validation."""
    
    from scipy.optimize import nnls
    
    print("Comparing CUDA PIRLS with scipy NNLS...")
    print("=" * 50)
    
    # Test problem
    n_samples = 256
    n_basis = 16
    n_pixels = 1000
    
    # Generate synthetic data
    np.random.seed(123)
    M = np.abs(np.random.randn(n_samples, n_basis))
    
    # Create ground truth with sparse structure
    true_beta = np.zeros((n_basis, n_pixels))
    for i in range(n_pixels):
        # Random 3-5 active components per pixel
        n_active = np.random.randint(3, 6)
        active_idx = np.random.choice(n_basis, n_active, replace=False)
        true_beta[active_idx, i] = np.abs(np.random.randn(n_active))
    
    # Generate noisy observations
    Y = M @ true_beta + 0.05 * np.random.randn(n_samples, n_pixels)
    Y = np.maximum(Y, 0)  # Ensure non-negative
    
    print(f"Problem size: {n_samples} samples × {n_basis} basis × {n_pixels} pixels")
    
    # Test scipy NNLS (on subset for speed)
    test_pixels = min(50, n_pixels)
    print(f"\nTesting scipy NNLS on {test_pixels} pixels...")
    
    start_time = time.time()
    beta_scipy = np.zeros((n_basis, test_pixels))
    for i in range(test_pixels):
        beta_scipy[:, i], _ = nnls(M, Y[:, i])
    scipy_time = time.time() - start_time
    
    print(f"Scipy NNLS: {scipy_time:.3f}s ({test_pixels/scipy_time:.1f} pixels/s)")
    
    # Test CUDA PIRLS
    if CUDA_AVAILABLE:
        print(f"\nTesting CUDA PIRLS on {n_pixels} pixels...")
        
        start_time = time.time()
        beta_cuda = pirls_batch_cuda(M, Y, max_iter=10)
        cuda_time = time.time() - start_time
        
        print(f"CUDA PIRLS: {cuda_time:.3f}s ({n_pixels/cuda_time:.1f} pixels/s)")
        
        # Compare accuracy on test subset
        mse_scipy = np.mean((beta_scipy - true_beta[:, :test_pixels]) ** 2)
        mse_cuda = np.mean((beta_cuda[:, :test_pixels] - true_beta[:, :test_pixels]) ** 2)
        
        print(f"\nAccuracy comparison (MSE):")
        print(f"Scipy NNLS: {mse_scipy:.6f}")
        print(f"CUDA PIRLS: {mse_cuda:.6f}")
        
        # Speedup
        est_scipy_total = scipy_time * (n_pixels / test_pixels)
        speedup = est_scipy_total / cuda_time
        print(f"\nEstimated speedup: {speedup:.1f}x")
        
    else:
        print("CUDA not available - skipping CUDA test")

def benchmark_scaling():
    """Benchmark CUDA PIRLS scaling with problem size."""
    
    if not CUDA_AVAILABLE:
        print("CUDA not available - skipping scaling benchmark")
        return
    
    print("\nCUDA PIRLS Scaling Benchmark")
    print("=" * 30)
    
    # Fixed parameters
    n_samples = 256
    n_basis = 16
    max_iter = 10
    
    # Different pixel counts
    pixel_counts = [100, 500, 1000, 2000, 5000, 10000]
    
    results = []
    
    for n_pixels in pixel_counts:
        print(f"\nTesting {n_pixels} pixels...")
        
        # Generate data
        M = np.abs(np.random.randn(n_samples, n_basis).astype(np.float32))
        Y = np.abs(np.random.randn(n_samples, n_pixels).astype(np.float32))
        
        # Benchmark
        start_time = time.time()
        beta = pirls_batch_cuda(M, Y, max_iter=max_iter)
        elapsed = time.time() - start_time
        
        speed = n_pixels / elapsed
        results.append((n_pixels, elapsed, speed))
        
        print(f"  Time: {elapsed:.3f}s, Speed: {speed:.0f} pixels/s")
    
    print(f"\nScaling Summary:")
    print(f"{'Pixels':<8} {'Time (s)':<10} {'Speed (px/s)':<12}")
    print("-" * 30)
    for n_pix, time_s, speed in results:
        print(f"{n_pix:<8} {time_s:<10.3f} {speed:<12.0f}")

def simple_test():
    """Simple test matching the original test_PIRLS_pycuda.py."""
    
    print("\nSimple Test (matching original)")
    print("=" * 30)
    
    # Simulated problem (same as original)
    n_samples = 512
    n_basis = 16
    n_pixels = 1024
    max_iter = 10

    M = np.abs(np.random.rand(n_samples, n_basis).astype(np.float32))
    Y = np.abs(np.random.rand(n_samples, n_pixels).astype(np.float32))

    if CUDA_AVAILABLE:
        start_time = time.time()
        Beta = pirls_batch_cuda(M, Y, max_iter=max_iter)
        elapsed = time.time() - start_time
        
        print("Beta shape:", Beta.shape)
        print("Sample Beta values:", Beta[:5, :5])
        print(f"Processing time: {elapsed:.3f}s")
        print(f"Speed: {n_pixels/elapsed:.0f} pixels/s")
    else:
        print("CUDA not available - cannot run test")

def main():
    """Run all tests."""
    
    print("PyCUDA PIRLS Test Suite")
    print("=" * 40)
    
    if not CUDA_AVAILABLE:
        print("❌ CUDA not available")
        print("Install PyCUDA: pip install pycuda")
        print("Ensure CUDA toolkit is installed")
        return
    
    print("✅ CUDA available")
    
    try:
        # Simple test (same as original)
        simple_test()
        
        # Basic validation
        compare_with_scipy()
        
        # Scaling benchmark
        benchmark_scaling()
        
        print("\n" + "=" * 40)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
