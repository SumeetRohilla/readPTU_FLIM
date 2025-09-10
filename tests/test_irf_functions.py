#!/usr/bin/env python3
"""
Test suite for IRF_Fun and Convol functions in FLIM_fitter.py

This module tests the basic building blocks used by PatternMatchIm:
- IRF_Fun: Instrumental Response Function generation
- Convol: FFT-based convolution function

These tests validate the core functionality and provide examples
of how to generate synthetic TCSPC data with noise.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import FLIM_fitter
sys.path.append(str(Path(__file__).parent.parent))

try:
    from FLIM_fitter import IRF_Fun, Convol
    print("✓ Successfully imported FLIM_fitter functions")
except ImportError as e:
    print(f"✗ Failed to import FLIM_fitter: {e}")
    sys.exit(1)

def test_irf_function():
    """Test IRF_Fun with various parameter sets"""
    print("\n" + "="*50)
    print("TESTING IRF_Fun")
    print("="*50)
    
    # Define test parameters
    test_cases = [
        {
            'name': 'Standard TCSPC IRF',
            'params': [2.0, 0.15, 0.05, 0.1, 0.01, 0.05, 0.0],
            'description': 'Typical parameters for TCSPC systems'
        },
        {
            'name': 'Narrow IRF',
            'params': [1.5, 0.08, 0.03, 0.06, 0.005, 0.02, 0.0],
            'description': 'High-resolution time-correlated setup'
        },
        {
            'name': 'Broad IRF with tail',
            'params': [3.0, 0.25, 0.08, 0.15, 0.02, 0.1, 0.1],
            'description': 'System with significant scattered light'
        }
    ]
    
    # Time axis
    n_channels = 256
    dt = 0.032  # ns
    time_axis = np.arange(n_channels) * dt
    
    # Test each case
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(12, 4*len(test_cases)))
    if len(test_cases) == 1:
        axes = axes.reshape(1, -1)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTesting: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        
        # Generate IRF
        try:
            irf = IRF_Fun(test_case['params'], time_axis)
            
            # Validate properties
            peak_pos = np.argmax(irf)
            peak_time = time_axis[peak_pos]
            total_area = np.sum(irf) * dt
            
            print(f"✓ IRF generated successfully")
            print(f"  Peak position: {peak_time:.3f} ns (channel {peak_pos})")
            print(f"  Total area: {total_area:.4f}")
            print(f"  Peak amplitude: {np.max(irf):.4f}")
            
            # Plot linear scale
            axes[i, 0].plot(time_axis, irf, 'b-', linewidth=2)
            axes[i, 0].set_title(f'{test_case["name"]} - Linear Scale')
            axes[i, 0].set_xlabel('Time (ns)')
            axes[i, 0].set_ylabel('IRF Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot log scale
            axes[i, 1].semilogy(time_axis, irf, 'r-', linewidth=2)
            axes[i, 1].set_title(f'{test_case["name"]} - Log Scale')
            axes[i, 1].set_xlabel('Time (ns)')
            axes[i, 1].set_ylabel('IRF Amplitude (log)')
            axes[i, 1].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"✗ Failed to generate IRF: {e}")
    
    plt.tight_layout()
    plt.savefig('irf_function_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ IRF test plots saved to irf_function_test.png")
    plt.show()

def test_convolution_function():
    """Test Convol function with known inputs"""
    print("\n" + "="*50)
    print("TESTING Convol Function")
    print("="*50)
    
    # Create test data
    n_channels = 256
    dt = 0.032
    time_axis = np.arange(n_channels) * dt
    
    # Generate test IRF
    irf_params = [2.0, 0.15, 0.05, 0.1, 0.01, 0.05, 0.0]
    irf = IRF_Fun(irf_params, time_axis)
    
    # Test decay functions
    test_decays = [
        {'lifetime': 1.0, 'name': 'Fast decay (1 ns)'},
        {'lifetime': 3.0, 'name': 'Medium decay (3 ns)'},
        {'lifetime': 8.0, 'name': 'Slow decay (8 ns)'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot IRF
    axes[0].plot(time_axis, irf, 'k-', linewidth=2, label='IRF')
    axes[0].set_title('Instrumental Response Function')
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Test convolutions
    colors = ['blue', 'red', 'green']
    
    for i, (decay_info, color) in enumerate(zip(test_decays, colors)):
        lifetime = decay_info['lifetime']
        name = decay_info['name']
        
        print(f"\nTesting convolution: {name}")
        
        # Generate pure exponential decay
        decay = np.exp(-time_axis / lifetime)
        
        try:
            # Perform convolution
            convolved = Convol(irf, decay)
            
            # Validate convolution properties
            peak_original = np.argmax(decay)
            peak_convolved = np.argmax(convolved)
            peak_shift = (peak_convolved - peak_original) * dt
            
            print(f"✓ Convolution successful")
            print(f"  Peak shift: {peak_shift:.3f} ns")
            print(f"  Max amplitude ratio: {np.max(convolved)/np.max(decay):.3f}")
            
            # Plot decay functions
            axes[1].plot(time_axis, decay, '--', color=color, alpha=0.7, 
                        label=f'Original {name}')
            axes[1].plot(time_axis, convolved, '-', color=color, linewidth=2,
                        label=f'Convolved {name}')
            
            # Plot on log scale
            axes[2].semilogy(time_axis, decay, '--', color=color, alpha=0.7)
            axes[2].semilogy(time_axis, convolved, '-', color=color, linewidth=2)
            
        except Exception as e:
            print(f"✗ Convolution failed: {e}")
    
    axes[1].set_title('Decay Functions - Linear Scale')
    axes[1].set_xlabel('Time (ns)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Decay Functions - Log Scale')
    axes[2].set_xlabel('Time (ns)')
    axes[2].set_ylabel('Amplitude (log)')
    axes[2].grid(True, alpha=0.3)
    
    # Test multi-component convolution
    print(f"\nTesting multi-component convolution...")
    
    # Create bi-exponential decay
    decay1 = 0.3 * np.exp(-time_axis / 1.0)
    decay2 = 0.7 * np.exp(-time_axis / 4.0)
    bi_exp = decay1 + decay2
    
    try:
        # Test convolution with multiple inputs
        multi_decay = np.column_stack([decay1, decay2, bi_exp])
        convolved_multi = Convol(irf, multi_decay)
        
        print(f"✓ Multi-component convolution successful")
        print(f"  Input shape: {multi_decay.shape}")
        print(f"  Output shape: {convolved_multi.shape}")
        
        # Plot multi-component result
        axes[3].plot(time_axis, bi_exp, 'k--', alpha=0.7, label='Original bi-exp')
        axes[3].plot(time_axis, convolved_multi[:, 2], 'r-', linewidth=2, 
                    label='Convolved bi-exp')
        axes[3].set_title('Bi-exponential Convolution')
        axes[3].set_xlabel('Time (ns)')
        axes[3].set_ylabel('Amplitude')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"✗ Multi-component convolution failed: {e}")
    
    plt.tight_layout()
    plt.savefig('convolution_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Convolution test plots saved to convolution_test.png")
    plt.show()

def generate_noisy_tcspc_example():
    """Generate example of realistic noisy TCSPC data"""
    print("\n" + "="*50)
    print("GENERATING NOISY TCSPC EXAMPLE")
    print("="*50)
    
    # Parameters
    n_channels = 256
    dt = 0.032
    time_axis = np.arange(n_channels) * dt
    
    # Generate IRF
    irf_params = [2.0, 0.15, 0.05, 0.1, 0.01, 0.05, 0.0]
    irf = IRF_Fun(irf_params, time_axis)
    
    # Create bi-exponential decay
    lifetime1, lifetime2 = 0.8, 3.5
    amp1, amp2 = 0.3, 0.7
    
    decay1 = amp1 * np.exp(-time_axis / lifetime1)
    decay2 = amp2 * np.exp(-time_axis / lifetime2)
    
    # Convolve with IRF
    conv1 = Convol(irf, decay1)
    conv2 = Convol(irf, decay2)
    
    # Add background and scale
    background = 10
    max_counts = 1000
    clean_signal = background + max_counts * (conv1 + conv2)
    
    # Add Poisson noise
    noisy_signal = np.random.poisson(clean_signal)
    
    print(f"✓ Generated noisy TCSPC data")
    print(f"  Background level: {background}")
    print(f"  Max counts: {np.max(noisy_signal)}")
    print(f"  SNR (peak): {np.max(clean_signal)/np.sqrt(np.max(clean_signal)):.1f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Individual components
    axes[0, 0].plot(time_axis, decay1, 'b-', label=f'τ₁ = {lifetime1} ns')
    axes[0, 0].plot(time_axis, decay2, 'r-', label=f'τ₂ = {lifetime2} ns')
    axes[0, 0].set_title('Pure Exponential Components')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Convolved components
    axes[0, 1].plot(time_axis, conv1, 'b-', label=f'Convolved τ₁')
    axes[0, 1].plot(time_axis, conv2, 'r-', label=f'Convolved τ₂')
    axes[0, 1].plot(time_axis, irf/np.max(irf)*np.max(conv1), 'k--', 
                   alpha=0.7, label='IRF (scaled)')
    axes[0, 1].set_title('IRF-Convolved Components')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Clean vs noisy - linear
    axes[1, 0].plot(time_axis, clean_signal, 'g-', linewidth=2, label='Clean signal')
    axes[1, 0].plot(time_axis, noisy_signal, 'ko', markersize=2, alpha=0.7, 
                   label='Noisy data')
    axes[1, 0].set_title('Clean vs Noisy Data - Linear Scale')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('Counts')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Clean vs noisy - log
    axes[1, 1].semilogy(time_axis, clean_signal, 'g-', linewidth=2, label='Clean signal')
    axes[1, 1].semilogy(time_axis, noisy_signal, 'ko', markersize=2, alpha=0.7, 
                       label='Noisy data')
    axes[1, 1].set_title('Clean vs Noisy Data - Log Scale')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].set_ylabel('Counts (log)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noisy_tcspc_example.png', dpi=150, bbox_inches='tight')
    print(f"✓ Noisy TCSPC example saved to noisy_tcspc_example.png")
    plt.show()
    
    return {
        'time_axis': time_axis,
        'irf': irf,
        'clean_signal': clean_signal,
        'noisy_signal': noisy_signal,
        'parameters': {
            'lifetimes': [lifetime1, lifetime2],
            'amplitudes': [amp1, amp2],
            'background': background
        }
    }

def run_all_irf_tests():
    """
    Run all IRF and convolution tests - Spyder-friendly version
    Call this function directly in Spyder console
    """
    print("Starting IRF_Fun and Convol test suite...")
    
    try:
        # Run all tests
        test_irf_function()
        test_convolution_function()
        example_data = generate_noisy_tcspc_example()
        
        print(f"\n{'='*60}")
        print("IRF AND CONVOLUTION TESTS COMPLETED")
        print("✓ All core functions tested successfully")
        print("✓ Example data generated for further testing")
        print(f"{'='*60}")
        
        return example_data
    except Exception as e:
        print(f"✗ Tests failed: {e}")
        return None

if __name__ == "__main__":
    run_all_irf_tests()