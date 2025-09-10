# readPTU_FLIM Library
### Advanced FLIM Analysis Pipeline for PicoQuant PTU files
**`Comprehensive toolkit for fluorescence lifetime imaging microscopy (FLIM) data analysis with deep learning-based cell segmentation.`**

- PicoQuant uses a bespoke file format called PTU to store data from time-tag-time-resolved (TTTR) measurements.<br/>
- Current file format (.ptu) supports both T2 and T3 acquisition modes for various TCSPC devices (MultiHarp, HydraHarp, TimeHarp, PicoHarp, etc.) <br/>
- At the moment, the library was tested for FLIM data obtained using MultiHarp, HydraHarp and PicoHarp. <br/>
- Includes advanced cell segmentation using PlantSeg and comprehensive fluorescence lifetime fitting capabilities.

### **`What is available`**
- ✅ PTU file reading and photon stream processing
- ✅ FLIM data analysis with multi-exponential fitting
- ✅ Cell segmentation using deep learning (U-Net) models via PlantSeg
- ✅ Distributed fluorescence lifetime analysis
- ✅ Parameter optimization for segmentation quality
- ✅ Comprehensive TCSPC analysis tools
- ✅ Support for PIE (Pulsed Interleaved Excitation) data
- ✅ Multi-frame and multi-channel analysis
- ✅ Automated batch processing with multiple output formats
- ✅ Professional visualization with CIM-style displays

## 🚀 Quick Start with Anaconda

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.8 or higher

### 1. Create and Activate Conda Environment

```bash
# Create a new conda environment
conda create -n ptu_flim python=3.9

# Activate the environment
conda activate ptu_flim
```

### 2. Install Core Dependencies

```bash
# Install from conda-forge for better compatibility
conda install -c conda-forge numpy matplotlib scipy numba scikit-learn pandas tqdm lmfit scikit-optimize tifffile
```

### 3. Install PlantSeg for Cell Segmentation (Required for CellsegPTU_FLIM.py only)

```bash
# Install PlantSeg dependencies
conda install -c conda-forge pytorch torchvision torchaudio
pip install plantseg
```

### 4. Alternative: Install All Dependencies at Once

Create an `environment.yml` file:

```yaml
name: ptu_flim
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - matplotlib
  - scipy
  - numba
  - scikit-learn
  - pandas
  - tqdm
  - lmfit
  - scikit-optimize
  - tifffile
  - pytorch
  - torchvision
  - torchaudio
  - pip
  - pip:
    - plantseg
    - pycuda  # Optional: for GPU acceleration
```

Then install:
```bash
conda env create -f environment.yml
conda activate ptu_flim
```

**Note**: PyCUDA requires NVIDIA CUDA Toolkit to be installed separately. See GPU Acceleration section below for details.

### 5. Verify Installation

Test your installation with:

```python
# Test core dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy
import numba
from sklearn.model_selection import ParameterGrid
import tqdm
import tifffile

# Test FLIM modules
from PTU_ScanRead import PTU_ScanRead, Process_Frame
from FLIM_fitter import FluoFit, DistFluoFit, Calc_mIRF

# Test PlantSeg (if installed)
try:
    from plantseg.predictions.functional.predictions import unet_predictions
    from plantseg.segmentation.functional.segmentation import mutex_ws
    print("✅ PlantSeg successfully imported")
except ImportError:
    print("⚠️ PlantSeg not available - CellsegPTU_FLIM.py features disabled")

print("✅ All core dependencies successfully imported!")
```

## 🚀 GPU Acceleration with CUDA (Optional)

### CUDA Requirements for GPU-Accelerated PIRLS

For significantly faster FLIM analysis using GPU acceleration, you can install CUDA support:

#### Tested Configuration
- **CUDA Toolkit**: 11.6
- **CUDA Compilation Tools**: Release 11.6, V11.6.55
- **Build**: cuda_11.6.r11.6/compiler.30794723_0

#### Installation Steps

1. **Install CUDA Toolkit 11.6**
   
   **Option 1: Using Conda (Recommended for Anaconda users)**
   ```bash
   # Activate your environment
   conda activate ptu_flim
   
   # Install CUDA toolkit 11.6 via conda
   conda install -c nvidia cuda-toolkit=11.6
   
   # Alternative: Install from conda-forge
   conda install -c conda-forge cudatoolkit=11.6
   ```
   
   **Option 2: Install NVIDIA drivers and CUDA separately**
   ```bash
   # Install NVIDIA drivers (if not already installed)
   conda install -c conda-forge nvidia-ml-py3
   
   # Install CUDA toolkit
   conda install -c nvidia cuda=11.6
   ```
   
   **Option 3: For systems without conda CUDA support**
   ```bash
   # Download and install from NVIDIA Developer site
   # https://developer.nvidia.com/cuda-11-6-0-download-archive
   # Follow platform-specific installation instructions
   ```

2. **Install PyCUDA**
   ```bash
   conda activate ptu_flim
   pip install pycuda
   ```

3. **Verify GPU Support**
   ```python
   # Test CUDA installation
   try:
       from FLIM_fitter import CUDA_AVAILABLE
       print(f"CUDA GPU acceleration: {'✅ Available' if CUDA_AVAILABLE else '❌ Not available'}")
   except ImportError:
       print("❌ CUDA support not installed")
   ```

#### Performance Benefits
- **10-50x speedup** for large FLIM datasets
- **Automatic fallback** to CPU if GPU unavailable
- **No code changes** required - acceleration is automatic

#### GPU Limitations
- Maximum 64 basis functions (lifetime components)
- Maximum 2048 time samples
- Requires NVIDIA GPU with compute capability 3.0+

#### Troubleshooting CUDA
- **"PyCUDA import failed"**: Check NVIDIA drivers and CUDA toolkit installation
- **"Kernel compilation failed"**: Verify GPU compute capability compatibility
- **"Problem size too large"**: Current GPU implementation has size limits
- **"Out of memory"**: Use windowed analysis or reduce batch size

##  Analysis Pipelines

The library provides two main analysis pipelines, each with different requirements and use cases:

### 🧬 **FlavMetaFLIM.py** - Metabolic FLIM Analysis

**Purpose**: Automated FLIM analysis for FAD metabolic imaging with batch processing capabilities.

**Key Features**:
- Automated batch processing of multiple folders
- Results saved next to raw PTU files
- Multiple output formats (.pkl, .csv, .tif, .png)
- CIM-style lifetime visualization
- Amplitude component analysis with scale bars
- Windowed vs pixel-wise analysis options

**Requirements**:
```bash
# Core scientific computing (required)
conda install -c conda-forge numpy matplotlib scipy numba scikit-learn pandas tqdm lmfit tifffile
```

**Usage**:
```python
from FlavMetaFLIM import process_single_file, process_multiple_folders

# Single file processing
results = process_single_file(
    'path/to/file.ptu',
    auto_det=0,          # Detector channel
    auto_PIE=1,          # PIE window
    flag_win=True,       # Use windowed analysis
    resolution=0.2,      # Time resolution (ns)
    tau0=[0.3, 1.7, 6.0] # Initial lifetime guesses
)

# Batch processing multiple folders
all_results = process_multiple_folders(
    ['folder1', 'folder2', 'folder3'],
    auto_det=0,
    flag_win=True,
    resolution=0.2
)
```

**Output Files** (saved next to PTU files):
- `filename_FLIM_results.pkl` - Complete analysis results
- `filename_lifetimes.csv` - Lifetime values by component
- `filename_amplitude_stats.csv` - Amplitude statistics
- `filename_intensity.tif` - Vignette-corrected intensity
- `filename_average_lifetime.tif` - Average lifetime map
- `filename_amplitude_*.tif` - Individual amplitude components
- `filename_FLIM_lifetime_cim.png` - CIM lifetime visualization
- `filename_amplitude_maps.png` - CIM amplitude subplots

### 🔬 **CellsegPTU_FLIM.py** - Cell Segmentation + FLIM Analysis

**Purpose**: Advanced cell segmentation using deep learning followed by cellular FLIM analysis.

**Key Features**:
- Deep learning-based cell segmentation (U-Net)
- Parameter optimization for segmentation quality
- Cell-by-cell lifetime analysis
- Multi-exponential fitting per cell
- Distributed lifetime fitting options

**Requirements**:
```bash
# Core dependencies + PlantSeg for segmentation
conda install -c conda-forge numpy matplotlib scipy numba scikit-learn pandas tqdm
conda install -c conda-forge pytorch torchvision torchaudio
pip install plantseg
```

**Usage**:
```python
# Edit the configuration section in CellsegPTU_FLIM.py:
filename = r'path/to/your/file.ptu'
mem_PIE = 2    # Membrane channel PIE window
mem_det = 2    # Membrane detector
auto_PIE = 1   # Autofluorescence PIE window  
auto_det = 1   # Autofluorescence detector

# Run the script
python CellsegPTU_FLIM.py
```

**Configuration Parameters**:
```python
# Channel assignments
mem_PIE = 2    # Laser pulse for membrane channel
mem_det = 2    # Detector for membrane channel
auto_PIE = 1   # Laser pulse for autofluorescence
auto_det = 1   # Detector for autofluorescence

# Analysis parameters
resolution = 0.2        # Temporal resolution (ns)
max_cell_Area = 5000   # Maximum cell area filter
cnum = 1               # PIE cycles (auto-detected)
```

**Output Variables**:
- `tauCell` - Cell-wise lifetime values [ncells x 3]
- `ACell` - Cell-wise amplitude coefficients [ncells x 3]
- `LIm` - Pixel-wise lifetime images [height x width x 3]
- `AIm` - Pixel-wise amplitude images [height x width x 3]
- `mask` - Final segmentation mask

## 📊 Supported Hardware

### TCSPC Devices
- **MultiHarp 150N/150P** (T2/T3 modes)
- **HydraHarp 400** (T2/T3 modes)
- **TimeHarp 260N/260P** (T2/T3 modes)
- **PicoHarp 300** (T2/T3 modes)

### Scanner Types
- Laser scanning microscopy (LSM)
- Piezo scanner systems
- Bidirectional and unidirectional scanning

## 📊 Test Data

**Test_FLIM_image_daisyPollen_PicoHarp_2.ptu**
- [Download test file](https://drive.google.com/file/d/1XtGL2yh_hJhaXIJhEDD5BpHNQXYQZX_p/view?usp=sharing)
- Acquired with: PicoHarp (T3 mode)
- Excitation: 485 nm laser
- Detection channels: 2
- Image: Daisy pollen FLIM data

![Interactive Demo Snapshot](Test_FLIM_image_daisyPollen_PicoHarp_2.png)

## 🔄 Recent Updates

- **2025**: **GPU acceleration** with CUDA PIRLS implementation (10-50x speedup)
- **2025**: Added cim-style amplitude visualization with subplot display and scale bars
- **2025**: Automated batch processing with results saved next to raw data
- **2025**: Multiple output formats (.pkl, .csv, .tif, .png)
- **2024**: Added comprehensive cell segmentation using PlantSeg U-Net models
- **2024**: Implemented distributed fluorescence lifetime fitting
- **2024**: Added PIE (Pulsed Interleaved Excitation) support
- **2024**: Multi-frame and multi-channel analysis capabilities
- **2024**: Parameter optimization for segmentation quality
- **2024**: Advanced FLIM fitting with maximum likelihood estimation

## 💡 Tips for Best Results

### For FlavMetaFLIM.py:
1. **Batch Processing**: Organize PTU files in separate folders for different conditions
2. **Parameter Tuning**: Adjust `tau0` initial guesses based on your sample
3. **Window Analysis**: Use `flag_win=True` for noisy data, `False` for high SNR
4. **File Organization**: Results automatically saved next to PTU files for easy access

### For CellsegPTU_FLIM.py:
1. **Memory Management**: For large PTU files, consider processing frame-by-frame
2. **Parameter Optimization**: Use parameter grid search for optimal segmentation
3. **Quality Control**: Filter cells by minimum photon count thresholds
4. **Visualization**: Always inspect segmentation results before analysis
5. **Channel Configuration**: Ensure correct PIE window and detector assignments

## 🐛 Troubleshooting

### Common Issues
- **PlantSeg Import Error**: Ensure PyTorch is properly installed before PlantSeg
- **Memory Issues**: Reduce image size or use windowed analysis for large datasets
- **Fitting Convergence**: Adjust initial parameter guesses (`tau0`)
- **Segmentation Quality**: Optimize beta and post_minsize parameters in CellsegPTU_FLIM.py
- **File Paths**: Use raw strings (r'path') or forward slashes for Windows paths

### Error Messages
- **"No photons found for channel X"**: Check channel assignments (auto_det, mem_det)
- **"Analysis failed"**: Verify PTU file integrity and parameter settings
- **PlantSeg model download**: Ensure internet connection for first-time model download

## 🤝 Contributing

Feel free to contribute improvements, bug fixes, or new features through pull requests!

## 📞 Support

Need help with MATLAB implementation? Email support is available!
