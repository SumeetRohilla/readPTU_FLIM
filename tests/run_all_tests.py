#!/usr/bin/env python3
"""
Test runner for all FLIM_fitter pattern matching tests

This script runs all available tests in the proper order:
1. Basic IRF and convolution tests
2. Pattern matching tests with different modes
3. Performance benchmarks (if GPU available)

Usage:
    python run_all_tests.py [--basic] [--gpu] [--plots]
    
Options:
    --basic     Run only basic CPU tests (no GPU requirements)
    --gpu       Run GPU-accelerated tests (requires CUDA)
    --plots     Save all plots to files
    --help      Show this help message
"""

import sys
import argparse
import time
from pathlib import Path

def setup_test_environment():
    """Set up test environment and check dependencies"""
    print("Setting up test environment...")
    
    # Check if we can import core modules
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from FLIM_fitter import IRF_Fun, Convol, PatternMatchIm
        print("✓ Core FLIM_fitter functions available")
        return True
    except ImportError as e:
        print(f"✗ Failed to import FLIM_fitter: {e}")
        return False

def check_gpu_availability():
    """Check if GPU acceleration is available"""
    try:
        import cupy as cp
        import numba.cuda as cuda
        if cuda.is_available():
            print("✓ GPU acceleration available")
            return True
        else:
            print("⚠ CUDA installed but no GPU detected")
            return False
    except ImportError:
        print("⚠ GPU acceleration not available (cupy/numba-cuda not installed)")
        return False

def run_basic_tests():
    """Run basic IRF and convolution tests"""
    print("\n" + "="*60)
    print("RUNNING BASIC TESTS")
    print("="*60)
    
    try:
        # Import and run IRF tests
        import test_irf_functions
        
        print("Running IRF function tests...")
        test_irf_functions.test_irf_function()
        
        print("Running convolution tests...")
        test_irf_functions.test_convolution_function()
        
        print("Generating noisy TCSPC example...")
        example_data = test_irf_functions.generate_noisy_tcspc_example()
        
        print("✓ Basic tests completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Basic tests failed: {e}")
        return False

def run_pattern_matching_tests(gpu_enabled=False):
    """Run pattern matching tests"""
    print("\n" + "="*60)
    print("RUNNING PATTERN MATCHING TESTS")
    print("="*60)
    
    try:
        import test_pattern_matching
        
        # Determine test configurations
        if gpu_enabled:
            test_configs = [
                ('small', 'single_exp'),
                ('small', 'bi_exp'),
                ('small', 'tri_exp'),
                ('large', 'bi_exp'),  # Large test only with GPU
            ]
        else:
            test_configs = [
                ('small', 'single_exp'),
                ('small', 'bi_exp'),
                ('small', 'tri_exp'),
            ]
        
        successful_tests = 0
        total_tests = len(test_configs)
        
        for test_size, decay_model in test_configs:
            try:
                print(f"\nRunning test: {test_size} size, {decay_model} model")
                summary = test_pattern_matching.run_comprehensive_test(test_size, decay_model)
                test_pattern_matching.print_test_summary(summary)
                successful_tests += 1
                
            except Exception as e:
                print(f"✗ Test failed ({test_size}, {decay_model}): {e}")
        
        print(f"\n✓ Pattern matching tests completed")
        print(f"  Successful: {successful_tests}/{total_tests}")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"✗ Pattern matching tests failed: {e}")
        return False

def run_performance_benchmark(gpu_enabled=False):
    """Run performance benchmark tests"""
    if not gpu_enabled:
        print("\n⚠ Skipping performance benchmark (GPU not available)")
        return True
    
    print("\n" + "="*60)
    print("RUNNING PERFORMANCE BENCHMARK")
    print("="*60)
    
    try:
        import test_pattern_matching
        
        # Benchmark different data sizes
        sizes = [
            (16, 16, "Small"),
            (32, 32, "Medium"),
            (64, 64, "Large"),
        ]
        
        results = []
        
        for nx, ny, size_name in sizes:
            print(f"\nBenchmarking {size_name} dataset ({nx}×{ny})...")
            
            # Create test data
            decay_model = test_pattern_matching.TestConfig.decay_models['bi_exp']
            data, ground_truth = test_pattern_matching.create_synthetic_tcspc_data(
                nx, ny, test_pattern_matching.TestConfig.n_channels, 
                decay_model, noise=True, spatial_variation=False
            )
            
            # Create basis matrix
            M = test_pattern_matching.create_basis_matrix(
                ground_truth['time_axis'], 
                ground_truth['lifetimes'], 
                ground_truth['irf']
            )
            
            # Test PIRLS mode performance
            result = test_pattern_matching.test_pattern_matching_modes(data, M, 'PIRLS')
            
            if result.get('success', False):
                results.append({
                    'size': size_name,
                    'dimensions': (nx, ny),
                    'pixels': nx * ny,
                    'performance': result['pixels_per_second'],
                    'time': result['elapsed_time']
                })
                print(f"  Performance: {result['pixels_per_second']:.1f} pixels/s")
            else:
                print(f"  Benchmark failed: {result.get('error', 'Unknown error')}")
        
        # Print benchmark summary
        if results:
            print(f"\nPerformance Benchmark Summary:")
            print(f"{'Size':<10} {'Pixels':<8} {'Time (s)':<10} {'Pixels/s':<12}")
            print("-" * 45)
            for r in results:
                print(f"{r['size']:<10} {r['pixels']:<8} {r['time']:<10.3f} {r['performance']:<12.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def main():
    """Main test runner function"""
    try:
        parser = argparse.ArgumentParser(description='Run FLIM_fitter pattern matching tests')
        parser.add_argument('--basic', action='store_true',
                           help='Run only basic CPU tests')
        parser.add_argument('--gpu', action='store_true',
                           help='Force GPU tests even if auto-detection fails')
        parser.add_argument('--plots', action='store_true',
                           help='Save all plots to files')
        parser.add_argument('--benchmark', action='store_true',
                           help='Run performance benchmarks')
        
        args = parser.parse_args()
    except SystemExit:
        # Fallback for Spyder - use default arguments
        print("⚠ Command line parsing failed, using default arguments")
        class Args:
            basic = False
            gpu = False
            plots = False
            benchmark = False
        args = Args()
    
    print("="*60)
    print("FLIM_FITTER PATTERN MATCHING TEST SUITE")
    print("="*60)
    print(f"Test runner started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup test environment
    if not setup_test_environment():
        print("✗ Test environment setup failed")
        return 1
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    if args.gpu and not gpu_available:
        print("⚠ GPU tests requested but GPU not available")
    
    gpu_enabled = gpu_available and not args.basic
    
    print(f"\nTest configuration:")
    print(f"  Basic tests: {'Yes' if not args.basic else 'Only'}")
    print(f"  GPU tests: {'Yes' if gpu_enabled else 'No'}")
    print(f"  Performance benchmark: {'Yes' if args.benchmark and gpu_enabled else 'No'}")
    print(f"  Save plots: {'Yes' if args.plots else 'No'}")
    
    # Run test suite
    start_time = time.time()
    test_results = []
    
    # 1. Basic tests
    basic_success = run_basic_tests()
    test_results.append(('Basic Tests', basic_success))
    
    # 2. Pattern matching tests
    if basic_success:
        pattern_success = run_pattern_matching_tests(gpu_enabled)
        test_results.append(('Pattern Matching', pattern_success))
        
        # 3. Performance benchmark
        if args.benchmark and gpu_enabled and pattern_success:
            benchmark_success = run_performance_benchmark(gpu_enabled)
            test_results.append(('Performance Benchmark', benchmark_success))
    
    # Print final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    print(f"Total execution time: {total_time:.1f} seconds")
    print(f"Test results:")
    
    all_passed = True
    for test_name, success in test_results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed successfully!")
        return_code = 0
    else:
        print(f"\n⚠ Some tests failed - check output above")
        return_code = 1
    
    print("="*60)
    return return_code

def run_tests_spyder_friendly(basic_only=False, enable_gpu=False, run_benchmark=False):
    """
    Spyder-friendly version of test runner that bypasses command line parsing
    
    Parameters
    ----------
    basic_only : bool
        Run only basic CPU tests
    enable_gpu : bool
        Force GPU tests even if auto-detection fails
    run_benchmark : bool
        Run performance benchmarks
    """
    print("="*60)
    print("FLIM_FITTER PATTERN MATCHING TEST SUITE (Spyder Mode)")
    print("="*60)
    print(f"Test runner started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup test environment
    if not setup_test_environment():
        print("✗ Test environment setup failed")
        return 1
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    if enable_gpu and not gpu_available:
        print("⚠ GPU tests requested but GPU not available")
    
    gpu_enabled = gpu_available and not basic_only
    
    print(f"\nTest configuration:")
    print(f"  Basic tests: {'Only' if basic_only else 'Yes'}")
    print(f"  GPU tests: {'Yes' if gpu_enabled else 'No'}")
    print(f"  Performance benchmark: {'Yes' if run_benchmark and gpu_enabled else 'No'}")
    
    # Run test suite
    start_time = time.time()
    test_results = []
    
    # 1. Basic tests
    basic_success = run_basic_tests()
    test_results.append(('Basic Tests', basic_success))
    
    # 2. Pattern matching tests
    if basic_success:
        pattern_success = run_pattern_matching_tests(gpu_enabled)
        test_results.append(('Pattern Matching', pattern_success))
        
        # 3. Performance benchmark
        if run_benchmark and gpu_enabled and pattern_success:
            benchmark_success = run_performance_benchmark(gpu_enabled)
            test_results.append(('Performance Benchmark', benchmark_success))
    
    # Print final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    print(f"Total execution time: {total_time:.1f} seconds")
    print(f"Test results:")
    
    all_passed = True
    for test_name, success in test_results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed successfully!")
        return_code = 0
    else:
        print(f"\n⚠ Some tests failed - check output above")
        return_code = 1
    
    print("="*60)
    return return_code

if __name__ == "__main__":
    # For Spyder compatibility - don't use sys.exit()
    result = main()
    print(f"Test runner completed with exit code: {result}")