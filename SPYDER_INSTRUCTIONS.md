# Running FLIM Tests in Spyder IDE

Here are several ways to run the FLIM pattern matching tests in Spyder:

## Method 1: Using the Spyder-Optimized Script (Recommended)

1. **Open** `run_tests_spyder.py` in Spyder
2. **Configure** the test settings at the top of the file:
   ```python
   TEST_MODE = 'basic'    # Change to 'gpu' or 'both'
   SHOW_PLOTS = True      # Set to False to suppress plots  
   SAVE_PLOTS = True      # Set to False to not save plots
   ```
3. **Run the entire file** by pressing `F5` or clicking the green "Run" button
4. The tests will execute and show results in the console

## Method 2: Running Individual Functions in Console

1. **First, run** `run_tests_spyder.py` once to load all functions
2. **Then in the Spyder console**, you can run individual tests:

```python
# Quick convolution test
quick_convol_test()

# Basic tests only
run_basic_tests_spyder()

# GPU tests (if available)
run_gpu_tests_spyder()
```

## Method 3: Step-by-Step Function Execution

Load and run specific test functions individually:

```python
# Import and test basic convolution
from test_convol_basic import test_convol_function
irf, decay, convolved = test_convol_function()

# Test multi-exponential models
from test_convol_basic import test_multi_exponential_convolution
results, irf, t_axis = test_multi_exponential_convolution()

# Test pattern matching
from test_convol_basic import test_pattern_matching_simple
coeffs, fit, M, lifetimes = test_pattern_matching_simple()
```

## Method 4: Direct Script Execution

1. **Open** either `test_convol_basic.py` or `test_gpu_pattern_matching.py` directly
2. **Run** the file with `F5` 
3. This will execute the `main()` function of that specific test

## Method 5: Console Commands

In the Spyder console, you can also run:

```python
# Execute any Python file directly
exec(open('test_convol_basic.py').read())

# Or run specific parts
exec(open('run_tests_spyder.py').read())
```

## Troubleshooting in Spyder

### If you get import errors:
```python
import sys
import os
sys.path.append(os.getcwd())  # Add current directory to path
```

### If plots don't show:
- Make sure Spyder's plot backend is set to "Inline" or "Automatic"
- Go to `Tools > Preferences > IPython console > Graphics > Backend`

### If GPU tests fail:
- Run basic tests only: `TEST_MODE = 'basic'`
- Or use: `run_basic_tests_spyder()`

### Memory issues:
- Restart the console: `Ctrl+.` (Ctrl + period)
- Or close plots: `plt.close('all')`

## Quick Start Example

1. Open `run_tests_spyder.py` in Spyder
2. Change the first line to: `TEST_MODE = 'basic'`  
3. Press `F5` to run
4. Watch the tests execute and plots appear

## Expected Output

You should see:
- Console output showing test progress
- Diagnostic plots in the Plots pane
- Saved PNG files in your working directory
- Test summary with success/failure status

## File Locations

Make sure these files are in your Spyder working directory:
- `FLIM_fitter.py` (your main module)
- `test_convol_basic.py`
- `test_gpu_pattern_matching.py` 
- `run_tests_spyder.py`

Check your working directory with: `os.getcwd()`