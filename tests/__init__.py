"""
FLIM Pattern Matching Test Suite

This package contains comprehensive tests for the PatternMatchIm function
and related FLIM analysis functions in FLIM_fitter.py.

Modules:
--------
test_pattern_matching : Main pattern matching test suite
test_irf_functions : IRF and convolution function tests  
run_all_tests : Comprehensive test runner

Usage:
------
# Run all tests
python tests/run_all_tests.py

# Run individual test modules
python tests/test_irf_functions.py
python tests/test_pattern_matching.py
"""

__version__ = "1.0.0"
__author__ = "FLIM Test Suite"

# Test configuration
TEST_CONFIG = {
    'small_image_size': (8, 8),
    'large_image_size': (32, 32),
    'time_channels': 256,
    'time_resolution': 0.032,  # ns
    'max_counts': 1000,
    'background_level': 10
}

# Export main test functions for easy access
try:
    from .test_pattern_matching import run_comprehensive_test
    from .test_irf_functions import test_irf_function, test_convolution_function
    from .run_all_tests import main as run_all_tests
    
    __all__ = [
        'run_comprehensive_test',
        'test_irf_function', 
        'test_convolution_function',
        'run_all_tests',
        'TEST_CONFIG'
    ]
    
except ImportError:
    # Handle relative imports when run directly
    __all__ = ['TEST_CONFIG']