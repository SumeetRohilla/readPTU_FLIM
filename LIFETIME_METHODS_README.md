# FLIM Lifetime Calculation Methods

This document describes the three lifetime calculation methods implemented in `FlavMetaFLIM.py` for analyzing FLIM data.

## Background

FLIM (Fluorescence Lifetime Imaging Microscopy) data contains multiple exponential components with different lifetimes (τ) and amplitudes (A). The challenge is to combine these into a single representative average lifetime value for each pixel.

## Implemented Methods

### 1. Amplitude-weighted Average Lifetime
**Variable**: `tau_avg_amplitude`

**Formula**: `τ_avg = Σ(A_i × τ_i)` (raw sum without normalization)

**Description**: Direct sum of amplitudes multiplied by their corresponding lifetimes. This is the raw amplitude-weighted sum that preserves the absolute contribution of each component.

### 2. Rate-based Average Lifetime
**Variable**: `tau_avg_rate_based`

**Formula**:
- `k_i = 1/τ_i` (convert lifetimes to rates)
- `k_avg = Σ(A_i × k_i) / Σ(A_i)`
- `τ_avg = 1/k_avg` (convert back to lifetime)

**Description**: First calculates the average decay rate (1/τ), then inverts to get the lifetime. This method emphasizes faster components more heavily and gives different results than arithmetic averaging because the harmonic mean is not equal to the arithmetic mean.

### 3. Variance-based Average Lifetime
**Variable**: `tau_var`

**Formula**: `τ_var = √[(Σ(t² × counts) / Σ(counts)) - (Σ(t × counts) / Σ(counts))²]`

**Description**: Calculates the variance (standard deviation) of photon arrival times directly from the TCSPC histogram. This provides a measure of the temporal spread of the fluorescence decay and represents the mean lifetime from the first moment of the decay curve.

## Physical Interpretation

- **Amplitude-weighted**: Raw weighted contribution of each component without normalization
- **Rate-based**: Emphasizes faster decay components, useful when decay rates are of primary interest
- **Variance-based**: Direct measure of temporal spread from the decay histogram

## Usage

All three methods are automatically calculated and saved when running FLIM analysis:

```python
results = analyze_flim_data(data, ...)

# Access the three lifetime methods
amplitude_weighted = results['tau_avg_amplitude']
rate_based = results['tau_avg_rate_based']
variance_based = results['tau_var']
```

## Output Files

The analysis saves the following files for each method:

### TIFF Images
- `*_rate_based_lifetime.tif`
- `*_variance_based_lifetime.tif`
- `*_average_lifetime.tif` (default method)

### Statistics CSV
- `*_lifetime_methods_stats.csv` - Comprehensive statistics for all methods

### Visualizations
- `*_lifetime_methods_comparison.png` - Side-by-side comparison of all methods

## Key Differences

1. **Amplitude-weighted vs Rate-based**:
   - Amplitude-weighted: Raw sum Σ(A_i × τ_i)
   - Rate-based: τ_avg = 1/[Σ(A_i × k_i) / Σ(A_i)] where k_i = 1/τ_i
   
2. **Mathematical relationship**: Rate-based typically gives shorter lifetimes than amplitude-weighted due to harmonic mean properties

3. **Variance-based**: Provides direct temporal measurement from histogram data

## When to Use Which Method

- **Amplitude-weighted**: When you need the raw weighted contribution without normalization
- **Rate-based**: When interested in average decay rates or when faster components should be emphasized
- **Variance-based**: When you want a direct measure of temporal spread from the decay data

## References

- Digman, M. A., et al. "The phasor approach to fluorescence lifetime imaging analysis." Biophysical journal 94.2 (2008): L14-L16.
- Lakowicz, J. R. "Principles of fluorescence spectroscopy." (2006).