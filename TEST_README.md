# FLIM Pattern Matching Test Suite

Comprehensive test suite for [`PatternMatchIm()`](FLIM_fitter.py:1499) function in [`FLIM_fitter.py`](FLIM_fitter.py) using synthetic TCSPC data generated with [`IRF_Fun()`](FLIM_fitter.py:108) and realistic noise models.

## Overview

This test suite validates the GPU-accelerated pattern matching functionality for fluorescence lifetime imaging microscopy (FLIM) data. It tests all three computational modes of [`PatternMatchIm()`](FLIM_fitter.py:1499) using synthetic time-correlated single photon counting (TCSPC) data with known ground truth parameters.

### Tested Functions
- **[`PatternMatchIm()`](FLIM_fitter.py:1499)** - Main pattern matching function with three modes
- **[`IRF_Fun()`](FLIM_fitter.py:108)** - Instrumental response function generation
- **[`Convol()`](FLIM_fitter.py:39)** - FFT-based convolution for IRF-decay modeling
- **[`PIRLSnonneg()`](FLIM_fitter.py:702)** - Poisson iterative reweighted least squares

## Files Structure

```
tests/
├── test_pattern_matching.py     # Main pattern matching test suite
├── test_irf_functions.py        # IRF and convolution function tests
├── run_all_tests.py             # Comprehensive test runner
└── [generated plots]            # Test result visualizations
```

### Test Scripts

#### [`tests/test_pattern_matching.py`](tests/test_pattern_matching.py)
- **Primary test suite** for [`PatternMatchIm()`](FLIM_fitter.py:1499) function
- Tests all three modes: `'Default'`, `'Nonneg'`, `'PIRLS'`
- Synthetic FLIM data generation with spatial lifetime variation
- Ground truth validation and performance benchmarking
- Comprehensive visualization of results

#### [`tests/test_irf_functions.py`](tests/test_irf_functions.py)
- **Core function validation** for [`IRF_Fun()`](FLIM_fitter.py:108) and [`Convol()`](FLIM_fitter.py:39)
- Multiple IRF parameter sets testing
- Convolution accuracy validation
- Noisy TCSPC data generation examples

#### [`tests/run_all_tests.py`](tests/run_all_tests.py)
- **Master test runner** with GPU detection
- Automated test sequencing
- Performance benchmarking
- Command-line interface for test configuration

## Quick Start

### Basic CPU-Only Tests
```bash
cd tests
python run_all_tests.py --basic
```

### Full GPU-Accelerated Tests
```bash
cd tests
python run_all_tests.py --gpu --benchmark
```

### Individual Test Modules
```bash
# Test core IRF functions only
python test_irf_functions.py

# Test pattern matching only
python test_pattern_matching.py
```

## Test Configuration

### Pattern Matching Modes

1. **`'Default'`** - Standard least squares fitting
   - Uses [`np.linalg.lstsq()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
   - Fast, unconstrained solution
   - Good for debugging and baseline performance

2. **`'Nonneg'`** - Non-negative least squares
   - Uses [`sklearn.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with positive constraints
   - Ensures physical meaningfulness (amplitudes ≥ 0)
   - Moderate computational cost

3. **`'PIRLS'`** - GPU-accelerated Poisson Iterative Reweighted Least Squares
   - Custom CUDA implementation for Poisson noise handling
   - Optimized for large FLIM datasets
   - Requires GPU with CUDA support

### Synthetic Data Parameters

```python
# Image dimensions
nx, ny = 32, 32          # Pixel array size
n_channels = 256         # Time histogram bins
dt = 0.032              # Time resolution (ns per channel)

# IRF parameters (realistic TCSPC values)
irf_params = [
    2.0,   # t_0: peak position (ns)
    0.15,  # w1: Gaussian width
    0.05,  # T1: first time constant
    0.1,   # T2: second time constant
    0.01,  # a: reconvolution amplitude
    0.05,  # b: tail amplitude
    0.0    # dt: time delay
]

# Decay models
single_exp: τ = [2.5] ns
bi_exp:     τ = [0.8, 3.5] ns, amplitudes = [0.3, 0.7]
tri_exp:    τ = [0.3, 1.8, 5.2] ns, amplitudes = [0.2, 0.5, 0.3]
```

## Test Features

### Synthetic Data Generation
- **IRF modeling**: Uses [`IRF_Fun()`](FLIM_fitter.py:108) with realistic TCSPC system parameters
- **Multi-exponential decays**: Single, bi-, and tri-exponential models
- **Spatial variation**: Smooth amplitude gradients across image pixels
- **Poisson noise**: Realistic photon counting statistics
- **Background levels**: Configurable constant offset

### Validation Metrics
- **Reconstruction accuracy**: Mean squared error between data and fit
- **Coefficient correlation**: Spatial amplitude map correlation with ground truth
- **Processing speed**: Pixels per second performance measurement
- **Memory efficiency**: GPU memory usage monitoring

### Visualization Outputs
- **Decay curve fitting**: Log-scale plots showing data vs. fitted curves
- **Spatial amplitude maps**: 2D visualization of lifetime component distributions
- **Residual analysis**: Spatial maps of fitting residuals
- **Performance benchmarks**: Timing and accuracy comparisons

## Expected Results

### Performance Benchmarks
- **Default mode**: ~50-100 pixels/second (CPU)
- **Nonneg mode**: ~20-50 pixels/second (CPU)  
- **PIRLS mode**: >500 pixels/second (GPU)

### Accuracy Targets
- **Reconstruction RMSE**: < 5% of signal amplitude
- **Amplitude correlation**: > 80% with ground truth
- **Lifetime recovery**: Within 10% of true values

### Generated Plots
```
tests/
├── irf_function_test.png           # IRF generation validation
├── convolution_test.png            # Convolution accuracy tests
├── noisy_tcspc_example.png         # Realistic noisy data example
├── pattern_matching_small_*.png    # Small dataset test results
└── pattern_matching_large_*.png    # Large dataset test results (GPU)
```

## Requirements

### Core Dependencies
```python
numpy          # Numerical operations
matplotlib     # Plotting and visualization  
scipy          # Optimization and signal processing
sklearn        # Non-negative least squares
```

### GPU Dependencies (Optional)
```python
cupy           # GPU array operations
numba          # CUDA kernel compilation
```

### System Requirements
- **CPU tests**: Any system with Python 3.7+
- **GPU tests**: NVIDIA GPU with CUDA capability ≥ 3.5
- **Memory**: 4+ GB RAM recommended for large datasets

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'FLIM_fitter'
```
**Solution**: Ensure [`FLIM_fitter.py`](FLIM_fitter.py) is in parent directory or Python path

#### 2. GPU Not Available
```
GPU support: Not available (cupy/numba-cuda not installed)
```
**Solution**: Run CPU-only tests with `--basic` flag or install CUDA dependencies

#### 3. Memory Errors
```
OutOfMemoryError: CUDA out of memory
```
**Solution**: Reduce test data size or use smaller image dimensions

#### 4. Slow Performance
```
PIRLS mode: 10 pixels/second (expected >500)
```
**Solutions**: 
- Check GPU utilization with `nvidia-smi`
- Verify CUDA installation
- Reduce batch size in PIRLS kernel

### Performance Notes
- PIRLS mode requires significant GPU memory for large datasets
- CPU modes are suitable for small datasets and algorithm validation
- Background processes can affect GPU performance measurements

## Validation Criteria

### Successful Test Indicators
1. **IRF Generation**: Smooth, normalized IRF with correct peak position
2. **Convolution**: Proper peak shift and shape preservation
3. **Pattern Matching**: Low reconstruction error (RMSE < 0.05)
4. **GPU Performance**: >100 pixels/second for PIRLS mode
5. **Coefficient Recovery**: >80% correlation with ground truth amplitudes

### Warning Signs
- High reconstruction errors (>10% RMSE)
- Poor correlation between fitted and true amplitudes
- Significantly degraded GPU performance
- NaN or infinite values in results
- Systematic residual patterns

## Customization

### Adding New Decay Models
```python
# In test_pattern_matching.py, modify TestConfig.decay_models:
new_model = {
    'lifetimes': [your_lifetimes],
    'amplitudes': [your_amplitudes], 
    'description': 'Custom decay model'
}
```

### Adjusting Test Parameters
```python
# Modify TestConfig class in test_pattern_matching.py:
class TestConfig:
    nx_large = 64           # Larger image size
    n_channels = 512        # Higher time resolution
    max_counts = 2000       # Higher signal levels
```

### Custom IRF Parameters
```python
# In test_irf_functions.py:
custom_irf_params = [
    peak_position,    # ns
    width,           # Gaussian width
    T1, T2,         # Time constants  
    a, b,           # Component amplitudes
    delay           # Time delay
]
```

## Output Interpretation

### Good Results
- Smooth fitted curves overlaying noisy experimental data
- Clear separation of lifetime components in spatial maps
- Low, randomly distributed residuals
- Consistent performance across different modes
- High correlation between recovered and true parameters

### Problematic Results  
- Poor fit quality in logarithmic scale plots
- Systematic patterns in residual maps
- Large discrepancies between modes
- Performance significantly below benchmarks
- Low correlation with ground truth

## Algorithm Validation

The test suite validates that [`PatternMatchIm()`](FLIM_fitter.py:1499) correctly implements:

1. **Linear least squares** (`'Default'` mode)
2. **Non-negative constraints** (`'Nonneg'` mode) 
3. **Poisson noise modeling** (`'PIRLS'` mode)

Each mode should recover known synthetic parameters within specified tolerances, providing confidence for analysis of real experimental data.

## References

- **FLIM Theory**: Becker, W. "Advanced Time-Correlated Single Photon Counting Techniques" (2005)
- **PIRLS Algorithm**: Green, P.J. "Iteratively reweighted least squares for maximum likelihood estimation" (1984)
- **GPU Acceleration**: Harris, M. "Optimizing CUDA" NVIDIA Developer Documentation

---

This comprehensive test suite ensures robust validation of the FLIM pattern matching pipeline and provides benchmarks for GPU acceleration performance in fluorescence lifetime analysis.