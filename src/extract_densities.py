#!/usr/bin/env python3
"""
Extract bootstrap densities from NWChem RT-TDDFT restart files

This script extracts the first N densities (typically 6: ρ₀ through ρ₅)
from NWChem restart files for use in bootstrapped LSTM predictions.

Usage:
    python extract_bootstrap_densities.py perm/prefix.rt_restart --n-densities 6
"""

import numpy as np
import argparse
import glob
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))
#from extract_single_dens import DensityExtractor
class DensityExtractor:
    """Extract density matrix from NWChem RT-TDDFT restart file"""
    
    def __init__(self, filename):
        """
        Initialize extractor
        
        Args:
            filename: Path to restart file
        """
        self.filename = filename
        self.metadata = {}
        
    def parse_header(self, lines):
        """
        Parse header information
        
        Args:
            lines: List of file lines
            
        Returns:
            Line number where data starts
        """
        data_start = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Parse metadata
            if line.startswith('nmats'):
                self.metadata['nmats'] = int(line.split()[1])
            elif line.startswith('nbf_ao'):
                self.metadata['nbf_ao'] = int(line.split()[1])
            elif line.startswith('it'):
                self.metadata['iteration'] = int(line.split()[1])
            elif line.startswith('t'):
                # Parse time in scientific notation
                time_str = line.split()[1]
                self.metadata['time'] = float(time_str.replace('E', 'e'))
            elif line.startswith('checksum'):
                checksum_str = line.split()[1]
                self.metadata['checksum'] = float(checksum_str.replace('E', 'e'))
                # Data starts on next line
                data_start = i + 1
                break
        
        if data_start is None:
            raise ValueError("Could not find data start (checksum line) in file")
        
        return data_start
    
    def values_to_complex_matrix(self, values, nbf):
        """
        Convert flat array of (real, imag) pairs to complex matrix
        
        Args:
            values: Array of alternating real and imaginary values
            nbf: Number of basis functions
            
        Returns:
            Complex nbf x nbf matrix
        """
        # Separate real and imaginary parts
        real_parts = values[0::2]  # Every even index
        imag_parts = values[1::2]  # Every odd index
        
        # Create complex array
        complex_values = real_parts + 1j * imag_parts
        
        # Reshape to matrix (stored column-major in Fortran)
        # NWChem stores matrices column-major
        matrix = complex_values.reshape((nbf, nbf), order='F')
        
        return matrix
    
    def parse_density_data(self, data_line):
        """
        Parse density matrix data from the data line
        
        Args:
            data_line: Line containing all density matrix elements
            
        Returns:
            (density_alpha, density_beta) as complex arrays
            If closed-shell (nmats=1), density_beta will be None
        """
        # Split the line into float values
        values = []
        for val_str in data_line.split():
            # Handle scientific notation with E or e
            val_str = val_str.replace('E', 'e')
            values.append(float(val_str))
        
        values = np.array(values)
        
        nbf = self.metadata['nbf_ao']
        nmats = self.metadata['nmats']
        
        # Each matrix element is stored as (real, imag) pair
        elements_per_matrix = nbf * nbf * 2  # 2 for real and imaginary
        
        if nmats == 1:
            # Closed-shell: only one density matrix
            alpha_values = values[:elements_per_matrix]
            beta_values = None
        elif nmats == 2:
            # Open-shell: alpha and beta densities
            alpha_values = values[:elements_per_matrix]
            beta_values = values[elements_per_matrix:2*elements_per_matrix]
        else:
            raise ValueError(f"Unexpected nmats={nmats}")
        
        # Convert to complex matrices
        density_alpha = self.values_to_complex_matrix(alpha_values, nbf)
        
        if beta_values is not None:
            density_beta = self.values_to_complex_matrix(beta_values, nbf)
        else:
            density_beta = None
        
        return density_alpha, density_beta
    
    def extract(self, spin='total'):
        """
        Extract density matrix from restart file
        
        Args:
            spin: Which density to extract
                  'alpha' - alpha spin density
                  'beta' - beta spin density
                  'total' - total density (alpha + beta)
                  
        Returns:
            density: Complex density matrix [nbf, nbf]
            metadata: Dictionary with file information
        """
        print(f"Reading file: {self.filename}")
        
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        data_start = self.parse_header(lines)
        
        print(f"\nFile metadata:")
        print(f"  Number of basis functions: {self.metadata['nbf_ao']}")
        print(f"  Number of matrices (nmats): {self.metadata['nmats']}")
        print(f"  Iteration: {self.metadata['iteration']}")
        print(f"  Time: {self.metadata['time']:.6f} a.u.")
        
        # Parse density data
        data_line = lines[data_start].strip()
        density_alpha, density_beta = self.parse_density_data(data_line)
        
        # Select requested density
        if spin == 'alpha':
            density = density_alpha
            print(f"\nExtracted: Alpha spin density")
        elif spin == 'beta':
            if density_beta is None:
                raise ValueError("Beta density not available (closed-shell calculation)")
            density = density_beta
            print(f"\nExtracted: Beta spin density")
        elif spin == 'total':
            if density_beta is not None:
                density = density_alpha + density_beta
                print(f"\nExtracted: Total density (alpha + beta)")
            else:
                density = density_alpha
                print(f"\nExtracted: Total density (closed-shell)")
        else:
            raise ValueError(f"Unknown spin option: {spin}")
        
        return density, self.metadata
    
    def validate_density(self, density):
        """
        Validate extracted density matrix
        
        Args:
            density: Complex density matrix
        """
        print(f"\nValidation:")
        print(f"  Shape: {density.shape}")
        print(f"  Dtype: {density.dtype}")
        
        # Check trace
        trace = np.trace(density).real
        print(f"  Trace (real): {trace:.10f}")
        print(f"  Trace (imag): {np.trace(density).imag:.2e}")
        
        # Check Hermiticity
        herm_error = np.linalg.norm(density - density.conj().T, 'fro')
        print(f"  Hermiticity error: ||ρ - ρ†|| = {herm_error:.2e}")
        
        if herm_error > 1e-6:
            print(f"  ⚠ Warning: Matrix is not Hermitian (error > 1e-6)")
        
        # Check eigenvalues (should be 0 ≤ λ ≤ 1 for idempotent density)
        eigenvalues = np.linalg.eigvalsh(density)
        print(f"  Eigenvalues: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
        
        if eigenvalues.min() < -1e-6:
            print(f"  ⚠ Warning: Negative eigenvalues detected")
        if eigenvalues.max() > 1.1:
            print(f"  ⚠ Warning: Large eigenvalues (> 1.1)")
        
        # Matrix norms
        frobenius_norm = np.linalg.norm(density, 'fro')
        print(f"  Frobenius norm: {frobenius_norm:.6f}")
        
        return {
            'trace': trace,
            'hermiticity_error': herm_error,
            'eigenvalues': (eigenvalues.min(), eigenvalues.max()),
            'frobenius_norm': frobenius_norm
        }

def find_restart_files(pattern):
    """
    Find restart files matching pattern
    
    Args:
        pattern: Glob pattern (e.g., "perm/h2_plus_rttddft.rt_restart")
        
    Returns:
        Sorted list of restart files
    """
    # Try with .* extension
    files = glob.glob(f"{pattern}.*")
    
    if not files:
        print(f"✗ No restart files found matching: {pattern}.*")
        return []
    
    # Sort by number
    def get_number(filename):
        try:
            return int(filename.split('.')[-1])
        except:
            return 0
    
    files.sort(key=get_number)
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Extract bootstrap densities from NWChem RT-TDDFT restart files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract first 6 densities (ρ₀ through ρ₅)
  python extract_bootstrap_densities.py perm/h2_plus_rttddft.rt_restart
  
  # Custom number of densities
  python extract_bootstrap_densities.py perm/h2_plus_rttddft.rt_restart --n-densities 10
  
  # Custom output prefix
  python extract_bootstrap_densities.py perm/h2_plus_rttddft.rt_restart --output rho
  
  # Different spin
  python extract_bootstrap_densities.py perm/h2_plus_rttddft.rt_restart --spin alpha

Output:
  Creates: rho_0.npy, rho_1.npy, rho_2.npy, ..., rho_5.npy

Usage in prediction:
  python src/predict_from_initial.py \\
      --model best_lstm_model.pt \\
      --bootstrap-densities rho_*.npy \\
      --field field.dat
  
  Or use JSON config with mode="bootstrap"
        """
    )
    
    parser.add_argument('restart_pattern', type=str,
                       help='Pattern for restart files (e.g., perm/h2_plus_rttddft.rt_restart)')
    parser.add_argument('--n-densities', type=int, default=6,
                       help='Number of densities to extract (default: 6 for history_length=5)')
    parser.add_argument('--output', type=str, default='rho',
                       help='Output filename prefix (default: rho → rho_0.npy, rho_1.npy, ...)')
    parser.add_argument('--spin', type=str, default='total',
                       choices=['alpha', 'beta', 'total'],
                       help='Which density to extract (default: total)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("="*70)
        print("EXTRACT BOOTSTRAP DENSITIES FROM RT-TDDFT")
        print("="*70)
        print(f"\nSearching for restart files: {args.restart_pattern}.*")
    
    # Find restart files
    files = find_restart_files(args.restart_pattern)
    
    if not files:
        print("\n✗ No restart files found!")
        sys.exit(1)
    
    print(f"Found {len(files)} restart files")
    
    if len(files) < args.n_densities:
        print(f"\n⚠ Warning: Requested {args.n_densities} densities but only {len(files)} files available")
        print(f"Will extract {len(files)} densities")
        args.n_densities = len(files)
    
    # Extract densities
    print(f"\nExtracting first {args.n_densities} densities...")
    
    densities = []
    output_files = []
    
    for i in range(args.n_densities):
        restart_file = files[i]
        output_file = f"{'densities/'+args.output}_{i}.npy"
        
        if not args.quiet:
            print(f"\n[{i+1}/{args.n_densities}] Processing: {restart_file}")
        
        # Extract density
        extractor = DensityExtractor(restart_file)
        density, metadata = extractor.extract(spin=args.spin)
        
        # Save
        np.save(output_file, density)
        densities.append(density)
        output_files.append(output_file)
        
        if not args.quiet:
            trace = np.trace(density).real
            time = metadata['time']
            print(f"  ✓ Saved: {output_file}")
            print(f"    Time: {time:.6f} a.u.")
            print(f"    Trace: {trace:.6f}")
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    
    print(f"\nExtracted {args.n_densities} densities:")
    for i, f in enumerate(output_files):
        trace = np.trace(densities[i]).real
        print(f"  ρ_{i}: {f} (Tr = {trace:.6f})")
    
    # Calculate time step
    if len(densities) >= 2:
        extractor0 = DensityExtractor(files[0])
        extractor0.parse_header(open(files[0]).readlines())
        t0 = extractor0.metadata['time']
        
        extractor1 = DensityExtractor(files[1])
        extractor1.parse_header(open(files[1]).readlines())
        t1 = extractor1.metadata['time']
        
        dt = t1 - t0
        print(f"\nTime step: Δt = {dt:.6f} a.u. ({dt*0.024189:.6f} fs)")
    
    # Print usage instructions
    print("\n" + "="*70)
    print("USAGE")
    print("="*70)
    
    print("\n1. Command-line prediction with bootstrapping:")
    print("\n2. For use with prediction JSON config:")
    print('       "bootstrap_files": [')
    for i, f in enumerate(output_files):
        comma = "," if i < len(output_files)-1 else ""
        print(f'         "{f}"{comma}')
    print('       ]')
    print('     },')
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
