# Using FLIM Pattern Matching Tests in Spyder

This guide shows how to run the [`PatternMatchIm()`](../FLIM_fitter.py:1499) tests directly in Spyder IDE.

## Quick Start

### 1. Import Test Functions
```python
# In Spyder console:
import sys
sys.path.append('tests')

# Import test modules
import test_pattern_matching
import test_irf_functions
import run_all_tests
```

### 2. Run Quick Demo (Recommended for first run)
```python
# Run a simple demonstration
result = test_pattern_matching.quick_demo()
```

### 3. Run All Tests (Spyder-Safe)
```python
# Run all tests with Spyder-friendly function
result = run_all_tests.run_tests_spyder_friendly()

# Or run with options:
result = run_all_tests.run_tests_spyder_friendly(basic_only=True)  # CPU only
result = run_all_tests.run_tests_spyder_friendly(enable_gpu=True, run_benchmark=True)  # Full tests
```

### 4. Run Individual Test Modules
```python
# Run pattern matching tests only
all_results = test_pattern_matching.run_all_pattern_tests()

# Run IRF and convolution tests only
example_data = test_irf_functions.run_all_irf_tests()
```

## Individual Test Functions

### Test Single Configuration
```python
# Test specific decay model and image size
summary = test_pattern_matching.run_comprehensive_test('small', 'bi_exp')
test_pattern_matching.print_test_summary(summary)
```

### Test Individual Components
```python
# Test IRF generation only
test_irf_functions.test_irf_function()

# Test convolution only  
test_irf_functions.test_convolution_function()

# Generate noisy example data
example = test_irf_functions.generate_noisy_tcspc_example()
```

## Available Test Configurations

### Image Sizes
- `'small'`: 8×8 pixels (fast testing)
- `'large'`: 32×32 pixels (requires more memory)

### Decay Models
- `'single_exp'`: Single exponential (τ = 2.5 ns)
- `'bi_exp'`: Bi-exponential (τ = [0.8, 3.5] ns)
- `'tri_exp'`: Tri-exponential (τ = [0.3, 1.8, 5.2] ns)

### Pattern Matching Modes
- `'Default'`: Standard least squares
- `'Nonneg'`: Non-negative least squares
- `'PIRLS'`: GPU-accelerated (requires CUDA)

## Expected Output

### Successful Test
```
✓ Successfully imported FLIM_fitter functions
✓ GPU support: Available
Creating synthetic data: 8×8×256
Decay model: Bi-exponential decay
✓ Generated data with max counts: 1045

--- Testing mode: Default ---
✓ Default completed successfully
  Time: 0.012 s
  Performance: 5333.3 pixels/s
  RMSE: 12.3456
  Relative error: 3.2%
```

### Test Summary
```
TEST SUMMARY
============================================================
Test configuration:
  Size: small
  Data shape: (8, 8, 256)
  Decay model: bi_exp

Results by mode:
  Default: SUCCESS - ✓ Valid
    Performance: 5333.3 pixels/s
    Accuracy: 0.945 correlation
  Nonneg: SUCCESS - ✓ Valid
    Performance: 2156.8 pixels/s
    Accuracy: 0.942 correlation
  PIRLS: SUCCESS - ✓ Valid
    Performance: 8642.1 pixels/s
    Accuracy: 0.948 correlation
```

## Troubleshooting

### Import Errors
```python
# If import fails, check FLIM_fitter.py location
import os
print("Current directory:", os.getcwd())
print("FLIM_fitter.py exists:", os.path.exists('FLIM_fitter.py'))
```

### GPU Issues
```python
# Check GPU availability
try:
    import cupy
    import numba.cuda as cuda
    print("CUDA available:", cuda.is_available())
    print("GPU count:", cuda.gpus)
except ImportError:
    print("GPU libraries not installed")
```

### Memory Issues
```python
# Use smaller test size
summary = test_pattern_matching.run_comprehensive_test('small', 'single_exp')
```

## Custom Testing

### Create Custom Test Data
```python
import numpy as np

# Custom parameters
nx, ny = 4, 4  # Very small for testing
n_channels = 128
decay_model = {
    'lifetimes': [1.0, 4.0],
    'amplitudes': [0.4, 0.6],
    'description': 'Custom bi-exponential'
}

# Generate data
data, ground_truth = test_pattern_matching.create_synthetic_tcspc_data(
    nx, ny, n_channels, decay_model, noise=True, spatial_variation=False
)

# Create basis matrix
M = test_pattern_matching.create_basis_matrix(
    ground_truth['time_axis'], 
    ground_truth['lifetimes'], 
    ground_truth['irf']
)

# Test specific mode
results = test_pattern_matching.test_pattern_matching_modes(data, M, 'Default')
print(f"Test completed: {results['success']}")
print(f"RMSE: {results.get('rmse', 'N/A')}")
```

### Visualize Results
```python
import matplotlib.pyplot as plt

# Plot test results
test_pattern_matching.plot_test_results(data, {'Default': results}, ground_truth)
plt.show()
```

## Performance Tips

1. **Start Small**: Use `'small'` image size and `'single_exp'` model for initial testing
2. **GPU Memory**: Close other applications if using PIRLS mode
3. **Plotting**: Set `save_path=None` in plot functions to avoid file I/O
4. **Multiple Runs**: Results may vary slightly due to Poisson noise

## Integration with FLIM Analysis

```python
# Example: Use test functions to validate real data processing
from FLIM_fitter import PatternMatchIm

# Your real FLIM data (nx, ny, n_channels)
real_data = your_flim_data

# Create basis from known lifetimes
lifetimes = [1.2, 3.8]  # Your expected lifetimes
M = test_pattern_matching.create_basis_matrix(time_axis, lifetimes, your_irf)

# Process with validated function
C, Z = PatternMatchIm(real_data, M, mode='Nonneg')

# Analyze results
print(f"Coefficient shape: {C.shape}")
print(f"Max amplitude: {np.max(C)}")
```

This approach ensures that [`PatternMatchIm()`](../FLIM_fitter.py:1499) is working correctly before applying it to experimental data.