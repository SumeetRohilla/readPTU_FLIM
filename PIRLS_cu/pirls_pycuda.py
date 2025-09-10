"""
PyCUDA implementation of PIRLS non-negative least squares solver
"""

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

# CUDA kernel code as string
kernel_code = """
#define MAX_BASIS 64
#define MAX_SAMPLES 2048
#define TINY 1e-6f
#define MAX_NNLS_ITER 100

__device__ void nnls_solve(float *A, float *b, float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = 0.0f;

    for (int iter = 0; iter < MAX_NNLS_ITER; iter++) {
        bool changed = false;
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                if (k != j) sum += A[j * n + k] * x[k];
            }
            float new_x = fmaxf(0.0f, (b[j] - sum) / fmaxf(A[j * n + j], TINY));
            if (fabsf(new_x - x[j]) > 1e-8f) changed = true;
            x[j] = new_x;
        }
        if (!changed) break;
    }
}

__global__ void PIRLS_nonneg_kernel(
    const float *M,
    const float *Y,
    float *Beta,
    int n_samples,
    int n_basis,
    int n_pixels,
    int max_iter) {

    int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= n_pixels) return;

    float beta[MAX_BASIS] = {0};
    float beta_new[MAX_BASIS] = {0};
    float w[MAX_SAMPLES];

    float MtM[MAX_BASIS * MAX_BASIS] = {0};
    float Mty[MAX_BASIS] = {0};

    for (int j = 0; j < n_basis; j++) {
        for (int k = 0; k < n_basis; k++) {
            float acc = 0.0f;
            for (int i = 0; i < n_samples; i++) {
                acc += M[i * n_basis + j] * M[i * n_basis + k];
            }
            MtM[j * n_basis + k] = acc;
        }
        float acc = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            acc += M[i * n_basis + j] * Y[i * n_pixels + pixel_id];
        }
        Mty[j] = acc;
    }

    nnls_solve(MtM, Mty, beta, n_basis);

    for (int iter = 0; iter < max_iter; ++iter) {
        float M_beta[MAX_SAMPLES];
        for (int i = 0; i < n_samples; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < n_basis; ++j) {
                dot += M[i * n_basis + j] * beta[j];
            }
            M_beta[i] = dot;
        }

        for (int i = 0; i < n_samples; ++i) {
            w[i] = 1.0f / fmaxf(M_beta[i], TINY);
        }

        float Aw[MAX_BASIS * MAX_BASIS] = {0};
        float bw[MAX_BASIS] = {0};

        for (int j = 0; j < n_basis; ++j) {
            for (int k = 0; k < n_basis; ++k) {
                float acc = 0.0f;
                for (int i = 0; i < n_samples; ++i) {
                    acc += M[i * n_basis + j] * w[i] * M[i * n_basis + k];
                }
                Aw[j * n_basis + k] = acc;
            }
            float acc = 0.0f;
            for (int i = 0; i < n_samples; ++i) {
                acc += M[i * n_basis + j] * w[i] * Y[i * n_pixels + pixel_id];
            }
            bw[j] = acc;
        }

        nnls_solve(Aw, bw, beta_new, n_basis);

        float err = 0.0f;
        for (int j = 0; j < n_basis; ++j) {
            float diff = beta_new[j] - beta[j];
            err += diff * diff;
            beta[j] = beta_new[j];
        }
        if (err < 1e-10f) break;
    }

    for (int j = 0; j < n_basis; ++j) {
        Beta[j * n_pixels + pixel_id] = beta[j];
    }
}
"""


# Compile the CUDA kernel
try:
    mod = SourceModule(kernel_code)
    kernel = mod.get_function("PIRLS_nonneg_kernel")
    CUDA_AVAILABLE = True
    print(" CUDA kernel compiled successfully")
except Exception as e:
    print(f"CUDA compilation failed: {e}")
    CUDA_AVAILABLE = False
    kernel = None

def run_pirls_pycuda(M, Y, n_samples, n_basis, n_pixels, max_iter=10):
    """
    Run PIRLS non-negative least squares on GPU using PyCUDA.
    
    Parameters
    ----------
    M : ndarray, shape (n_samples, n_basis), dtype=float32
        Design matrix
    Y : ndarray, shape (n_samples, n_pixels), dtype=float32
        Data matrix
    n_samples : int
        Number of samples (time points)
    n_basis : int
        Number of basis functions
    n_pixels : int
        Number of pixels to process
    max_iter : int
        Maximum PIRLS iterations
        
    Returns
    -------
    Beta : ndarray, shape (n_basis, n_pixels), dtype=float32
        Fitted coefficients
    """

    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernel not available - check PyCUDA installation")

    # Validate input dimensions
    assert M.shape == (n_samples, n_basis), f"M shape mismatch: {M.shape} vs ({n_samples}, {n_basis})"
    assert Y.shape == (n_samples, n_pixels), f"Y shape mismatch: {Y.shape} vs ({n_samples}, {n_pixels})"
    assert M.dtype == np.float32, f"M must be float32, got {M.dtype}"
    assert Y.dtype == np.float32, f"Y must be float32, got {Y.dtype}"

    # Check size limits
    if n_basis > 64:
        raise ValueError(f"n_basis ({n_basis}) exceeds maximum (64)")
    if n_samples > 2048:
        raise ValueError(f"n_samples ({n_samples}) exceeds maximum (2048)")

    # Allocate GPU memory
    M_gpu = cuda.mem_alloc(M.nbytes)
    Y_gpu = cuda.mem_alloc(Y.nbytes)
    Beta = np.zeros((n_basis, n_pixels), dtype=np.float32)
    Beta_gpu = cuda.mem_alloc(Beta.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(M_gpu, M)
    cuda.memcpy_htod(Y_gpu, Y)

    # Launch kernel
    threads_per_block = 64
    blocks_per_grid = (n_pixels + threads_per_block - 1) // threads_per_block

    kernel(M_gpu, Y_gpu, Beta_gpu,
        np.int32(n_samples), np.int32(n_basis), np.int32(n_pixels), np.int32(max_iter),
        block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1, 1))

    # Copy result back to host
    cuda.memcpy_dtoh(Beta, Beta_gpu)

    # Clean up GPU memory
    M_gpu.free()
    Y_gpu.free()
    Beta_gpu.free()

    return Beta


def pirls_batch_cuda(M, Y, max_iter=10):
    """
    Convenient wrapper that handles data conversion and reshaping.
    
    Parameters
    ----------
    M : ndarray, shape (n_samples, n_basis)
        Design matrix (any dtype, will be converted to float32)
    Y : ndarray, shape (n_samples, n_pixels)
        Data matrix (any dtype, will be converted to float32)
    max_iter : int
        Maximum PIRLS iterations
        
    Returns
    -------
    Beta : ndarray, shape (n_basis, n_pixels), dtype=float32
        Fitted coefficients
    """
    
    # Convert to required format
    M_f32 = np.ascontiguousarray(M, dtype=np.float32)
    Y_f32 = np.ascontiguousarray(Y, dtype=np.float32)
    
    n_samples, n_basis = M_f32.shape
    n_pixels = Y_f32.shape[1]
    
    return run_pirls_pycuda(M_f32, Y_f32, n_samples, n_basis, n_pixels, max_iter)

# Test function
def test_pirls_cuda():
    """Simple test of CUDA PIRLS implementation."""
    
    if not CUDA_AVAILABLE:
        print("CUDA not available - skipping test")
    return

    print("Testing Fixed CUDA PIRLS implementation...")

    # Test problem
    n_samples = 256
    n_basis = 8
    n_pixels = 100

    # Generate test data that matches the CPU test
    np.random.seed(42)
    M = np.abs(np.random.randn(n_samples, n_basis).astype(np.float32))
    true_beta = np.abs(np.random.randn(n_basis, n_pixels).astype(np.float32))
    Y = M @ true_beta + 0.1 * np.abs(np.random.randn(n_samples, n_pixels).astype(np.float32))

    # Test CPU version for comparison
    try:
        from FLIM_fitter import PIRLSnonneg_batch
        print("Running CPU PIRLS...")
        import time
        start = time.time()
        Beta_cpu = PIRLSnonneg_batch(M, Y, max_iter=10)
        cpu_time = time.time() - start
        
        # Test CUDA version
        print("Running CUDA PIRLS...")
        start = time.time()
        Beta_cuda = pirls_batch_cuda(M, Y, max_iter=10)
        cuda_time = time.time() - start
        
        # Compare results
        mse = np.mean((Beta_cpu - Beta_cuda) ** 2)
        max_diff = np.max(np.abs(Beta_cpu - Beta_cuda))
        
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"CUDA time: {cuda_time:.3f}s")
        print(f"Speedup: {cpu_time/cuda_time:.2f}x")
        print(f"MSE difference: {mse:.8f}")
        print(f"Max difference: {max_diff:.8f}")
        print(f"CUDA Results - Min: {np.min(Beta_cuda):.6f}, Max: {np.max(Beta_cuda):.6f}")
        print(f"CPU Results  - Min: {np.min(Beta_cpu):.6f}, Max: {np.max(Beta_cpu):.6f}")
        
        if mse < 1e-6:
            print("✅ CUDA implementation matches CPU results!")
        else:
            print("❌ Results don't match - need further debugging")
            
    except ImportError:
        print("PIRLSnonneg_batch not available for comparison")
        # Just test CUDA version
        import time
        start_time = time.time()
        Beta_cuda = pirls_batch_cuda(M, Y, max_iter=10)
        elapsed = time.time() - start_time
        
        print(f"CUDA test completed in {elapsed:.3f} seconds")
        print(f"Speed: {n_pixels/elapsed:.0f} pixels/second")
        print(f"Beta range: [{np.min(Beta_cuda):.3f}, {np.max(Beta_cuda):.3f}]")
        
        # Check for zeros (the original problem)
        if np.all(Beta_cuda == 0):
            print("❌ Still getting all zeros - algorithm needs more debugging")
        else:
            print("✅ Getting non-zero results!")

    return Beta_cuda


if __name__ == "__main__":
    test_pirls_cuda()