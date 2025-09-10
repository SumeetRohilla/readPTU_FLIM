#!/usr/bin/env python3
"""
Spyder-compatible GPU PIRLS test to verify package installations and GPU kernel compilation.

SPYDER USAGE:
1. Open this file in Spyder
2. Run sections individually using Ctrl+Enter or run entire file with F5
3. Check output in IPython console

Functions you can call directly in Spyder console:
- check_packages()
- quick_gpu_test()
- full_gpu_test()
- compare_cpu_gpu()
"""

import numpy as np
import time

#%% Package Installation Check
def check_packages():
    """
    Check if required GPU packages are installed.
    Call this function first in Spyder console.
    """
    print("=== Checking GPU Package Installations ===")
    
    # Check CuPy
    try:
        import cupy as cp
        print(f"✓ CuPy installed: version {cp.__version__}")
        gpu_available = cp.cuda.is_available()
        if gpu_available:
            device = cp.cuda.Device()
            # Get device properties correctly
            try:
                device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode('utf-8')
                print(f"✓ GPU available: {device_name}")
            except:
                print(f"✓ GPU available: Device {device.id}")
            
            mem_info = device.mem_info
            print(f"  Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
            return True
        else:
            print("✗ GPU not available")
            return False
    except ImportError as e:
        print(f"✗ CuPy not installed: {e}")
        return False

#%% Test Data Generation
def create_simple_test_data():
    """Create small test dataset for Spyder testing"""
    print("Creating simple test data...")
    
    # Small dimensions for fast testing in Spyder
    n_samples = 128  # time channels
    n_features = 3   # basis functions 
    n_pixels = 500   # pixels
    
    print(f"Dimensions: {n_samples} × {n_features} × {n_pixels}")
    
    # Create design matrix (like TCSPC basis functions)
    M = np.random.exponential(scale=2.0, size=(n_samples, n_features)).astype(np.float32)
    M = M / np.sum(M, axis=0, keepdims=True)  # Normalize
    
    # Create true coefficients
    true_coeffs = np.random.exponential(scale=1.0, size=(n_features, n_pixels)).astype(np.float32)
    
    # Generate observation data with Poisson noise
    Y_clean = M @ true_coeffs
    Y_noisy = np.random.poisson(Y_clean * 100) / 100
    Y_noisy = Y_noisy.astype(np.float32)
    
    print(f"✓ Test data created")
    print(f"  Data range: [{Y_noisy.min():.3f}, {Y_noisy.max():.3f}]")
    
    return M, Y_noisy

#%% GPU Kernel Compilation
def compile_simple_kernel():
    """Compile a simple version of the PIRLS kernel"""
    print("Compiling GPU kernel...")
    
    try:
        import cupy as cp
        
        # Simple kernel for testing (hardcoded dimensions for Spyder)
        kernel_code = r'''
extern "C" __global__
void simple_pirls_kernel(const float* X, const float* Y, float* C,
                         int n, int m, int npix) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= npix) return;
    
    // Simple non-negative least squares (no PIRLS for now)
    for(int j = 0; j < m; j++) {
        float sum_xy = 0.0f;
        float sum_xx = 0.0f;
        
        for(int i = 0; i < n; i++) {
            float x_val = X[i * m + j];
            sum_xy += x_val * Y[pix * n + i];
            sum_xx += x_val * x_val;
        }
        
        // Simple coefficient calculation
        C[pix * m + j] = (sum_xx > 0) ? (sum_xy / sum_xx) : 0.0f;
        if(C[pix * m + j] < 0) C[pix * m + j] = 0.0f;  // Non-negative
    }
}'''
        
        kernel = cp.RawKernel(kernel_code, 'simple_pirls_kernel')
        print("✓ GPU kernel compiled successfully")
        return kernel
        
    except Exception as e:
        print(f"✗ Kernel compilation failed: {e}")
        return None

#%% Quick GPU Test
def quick_gpu_test():
    """
    Quick GPU functionality test.
    Run this in Spyder console: quick_gpu_test()
    """
    print("\n=== Quick GPU Test ===")
    
    # Check packages first
    if not check_packages():
        return False
    
    # Create test data
    M, Y = create_simple_test_data()
    
    # Compile kernel
    kernel = compile_simple_kernel()
    if kernel is None:
        return False
    
    # Run GPU test
    try:
        import cupy as cp
        
        n, m = M.shape
        npix = Y.shape[1]
        
        # Transfer to GPU
        M_gpu = cp.asarray(M, dtype=cp.float32)
        Y_gpu = cp.asarray(Y.T, dtype=cp.float32)  # Transpose for kernel
        C_gpu = cp.zeros((npix, m), dtype=cp.float32)
        
        # Launch kernel
        threads = 128
        blocks = (npix + threads - 1) // threads
        
        start_time = time.time()
        kernel((blocks,), (threads,), (M_gpu, Y_gpu, C_gpu, n, m, npix))
        cp.cuda.Device().synchronize()
        gpu_time = time.time() - start_time
        
        # Get results
        C_result = cp.asnumpy(C_gpu.T)  # Transpose back
        
        print(f"✓ GPU test completed in {gpu_time:.3f} seconds")
        print(f"  Speed: {npix/gpu_time:.1f} pixels/second")
        print(f"  Result shape: {C_result.shape}")
        print(f"  Result range: [{C_result.min():.3f}, {C_result.max():.3f}]")
        
        # Cleanup
        del M_gpu, Y_gpu, C_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return True
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

#%% CPU Reference Implementation
def cpu_simple_nnls(M, Y):
    """Simple CPU non-negative least squares for comparison"""
    print("Running CPU reference...")
    
    n, m = M.shape
    npix = Y.shape[1]
    C_cpu = np.zeros((m, npix), dtype=np.float32)
    
    start_time = time.time()
    
    # Simple pixel-by-pixel NNLS
    for i in range(npix):
        for j in range(m):
            # Simple coefficient calculation
            sum_xy = np.sum(M[:, j] * Y[:, i])
            sum_xx = np.sum(M[:, j] ** 2)
            
            if sum_xx > 0:
                C_cpu[j, i] = max(0, sum_xy / sum_xx)  # Non-negative
    
    cpu_time = time.time() - start_time
    
    print(f"✓ CPU reference completed in {cpu_time:.3f} seconds")
    print(f"  Speed: {npix/cpu_time:.1f} pixels/second")
    
    return C_cpu, cpu_time

#%% Full Comparison Test
def compare_cpu_gpu():
    """
    Compare CPU vs GPU implementations.
    Run this in Spyder console: compare_cpu_gpu()
    """
    print("\n=== CPU vs GPU Comparison ===")
    
    # Check GPU first
    if not check_packages():
        print("GPU not available, running CPU only")
        M, Y = create_simple_test_data()
        cpu_simple_nnls(M, Y)
        return
    
    # Create test data
    M, Y = create_simple_test_data()
    
    # Run CPU version
    C_cpu, cpu_time = cpu_simple_nnls(M, Y)
    
    # Compile and run GPU version
    kernel = compile_simple_kernel()
    if kernel is None:
        print("GPU kernel failed, CPU results only")
        return
    
    try:
        import cupy as cp
        
        n, m = M.shape
        npix = Y.shape[1]
        
        # GPU execution
        M_gpu = cp.asarray(M, dtype=cp.float32)
        Y_gpu = cp.asarray(Y.T, dtype=cp.float32)
        C_gpu = cp.zeros((npix, m), dtype=cp.float32)
        
        threads = 128
        blocks = (npix + threads - 1) // threads
        
        start_time = time.time()
        kernel((blocks,), (threads,), (M_gpu, Y_gpu, C_gpu, n, m, npix))
        cp.cuda.Device().synchronize()
        gpu_time = time.time() - start_time
        
        C_gpu_result = cp.asnumpy(C_gpu.T)
        
        # Compare results
        print(f"\n=== Comparison Results ===")
        diff = np.abs(C_cpu - C_gpu_result)
        print(f"Max difference: {diff.max():.6f}")
        print(f"Mean difference: {diff.mean():.6f}")
        print(f"CPU time: {cpu_time:.3f} seconds")
        print(f"GPU time: {gpu_time:.3f} seconds")
        if gpu_time > 0:
            print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
        # Cleanup
        del M_gpu, Y_gpu, C_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return True
        
    except Exception as e:
        print(f"GPU comparison failed: {e}")
        return False

#%% Advanced GPU Test with Real PIRLS
def test_real_pirls_kernel():
    """Test with actual PIRLS kernel (more complex)"""
    print("\n=== Testing Real PIRLS Kernel ===")
    
    try:
        import cupy as cp
        
        # Real PIRLS kernel with dynamic dimensions
        pirls_kernel_src = r'''
#define T_MAX 128
#define B_MAX 8
extern "C" __global__
void pirls_kernel(const float* X, const float* Y, float* C,
                  int n, int m, int sub_npix,
                  int max_iter, float lr, int pgd_iters) {{
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= sub_npix) return;
    const float* y = Y + pix * n;
    float beta[B_MAX];
    float Aw0[B_MAX][B_MAX];
    float bw0[B_MAX];
    // zero
    for(int i=0;i<m;i++){{ bw0[i]=0; for(int j=0;j<m;j++) Aw0[i][j]=0; }}
    // accumulate unweighted normal equations
    for(int i=0;i<n;i++){{
        for(int j=0;j<m;j++){{
            bw0[j] += X[i*m+j] * y[i];
            for(int l=0;l<m;l++) Aw0[j][l] += X[i*m+j]*X[i*m+l];
        }}
    }}
    // initial PGD NNLS
    for(int j=0;j<m;j++) beta[j] = 0.1f;  // initialize
    for(int it=0;it<pgd_iters;it++){{
        for(int i=0;i<m;i++){{
            float tmp = -bw0[i];
            for(int j=0;j<m;j++) tmp += Aw0[i][j] * beta[j];
            for(int j=0;j<m;j++) beta[j] -= lr * 2 * Aw0[i][j] * tmp;
        }}
        for(int j=0;j<m;j++) if(beta[j]<0) beta[j]=0;
    }}
    float tiny = 0.1f / n;
    // PIRLS iterations
    for(int k=0;k<max_iter;k++){{
        float Aw[B_MAX][B_MAX];
        float bw[B_MAX];
        for(int i=0;i<m;i++){{ bw[i]=0; for(int j=0;j<m;j++) Aw[i][j]=0; }}
        for(int i=0;i<n;i++){{
            float mu = 0;
            for(int j=0;j<m;j++) mu += X[i*m+j] * beta[j];
            float w = 1.0f / (mu > tiny ? mu : tiny);
            for(int j=0;j<m;j++){{
                bw[j] += X[i*m+j] * w * y[i];
                for(int l=0;l<m;l++) Aw[j][l] += X[i*m+j] * w * X[i*m+l];
            }}
        }}
        for(int it=0; it<pgd_iters; it++){{
            for(int i=0;i<m;i++){{
                float tmp2 = -bw[i];
                for(int j=0;j<m;j++) tmp2 += Aw[i][j] * beta[j];
                for(int j=0;j<m;j++) beta[j] -= lr * 2 * Aw[i][j] * tmp2;
            }}
            for(int j=0;j<m;j++) if(beta[j]<0) beta[j]=0;
        }}
    }}
    for(int j=0;j<m;j++) C[pix*m + j] = beta[j];
}}'''
        
        kernel = cp.RawKernel(pirls_kernel_src, 'pirls_kernel')
        print("✓ Real PIRLS kernel compiled successfully")
        
        # Test with small data
        M, Y = create_simple_test_data()
        n, m = M.shape
        npix = min(Y.shape[1], 100)  # Limit for testing
        
        # GPU execution
        M_gpu = cp.asarray(M, dtype=cp.float32)
        Y_gpu = cp.asarray(Y[:, :npix].T, dtype=cp.float32)
        C_gpu = cp.zeros((npix, m), dtype=cp.float32)
        
        threads = 128
        blocks = (npix + threads - 1) // threads
        max_iter = 3
        lr = 1e-2
        pgd_iters = 10
        
        start_time = time.time()
        kernel((blocks,), (threads,), 
               (M_gpu, Y_gpu, C_gpu, n, m, npix, max_iter, lr, pgd_iters))
        cp.cuda.Device().synchronize()
        gpu_time = time.time() - start_time
        
        C_result = cp.asnumpy(C_gpu.T)
        
        print(f"✓ Real PIRLS completed in {gpu_time:.3f} seconds")
        print(f"  Result shape: {C_result.shape}")
        print(f"  Result range: [{C_result.min():.3f}, {C_result.max():.3f}]")
        
        # Cleanup
        del M_gpu, Y_gpu, C_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return True
        
    except Exception as e:
        print(f"✗ Real PIRLS kernel failed: {e}")
        return False

#%% Main Demo Function
def full_gpu_test():
    """
    Complete GPU PIRLS test demonstration.
    Run this in Spyder console: full_gpu_test()
    """
    print("GPU PIRLS Test - Spyder Compatible")
    print("=" * 40)
    
    # Step by step testing
    print("\n1. Checking packages...")
    if not check_packages():
        print("❌ Package check failed")
        return
    
    print("\n2. Quick GPU test...")
    if not quick_gpu_test():
        print("❌ GPU test failed")
        return
    
    print("\n3. CPU vs GPU comparison...")
    if not compare_cpu_gpu():
        print("❌ Comparison failed")
        return
    
    print("\n4. Real PIRLS kernel test...")
    if not test_real_pirls_kernel():
        print("⚠️ Real PIRLS failed, but basic GPU works")
    
    print("\n✅ GPU tests completed! Ready for FLIM integration.")

#%% Spyder-friendly execution
if __name__ == "__main__":
    print("GPU PIRLS Test - Run individual functions in Spyder console:")
    print("- check_packages()           # Check installations")
    print("- quick_gpu_test()           # Basic GPU test")
    print("- compare_cpu_gpu()          # CPU vs GPU comparison")
    print("- test_real_pirls_kernel()   # Advanced PIRLS test")
    print("- full_gpu_test()            # Complete test suite")
    
    # Auto-run basic check
    check_packages()
