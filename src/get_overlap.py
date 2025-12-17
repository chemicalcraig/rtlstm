#!/usr/bin/env python3
"""
Parse overlap matrix from modified NWChem output

Format:
 Overlap matrix
 --------------
         1         1    0.1000000000E+01
         1         2    0.6458989405E+00
         ...

Usage:
    python parse_custom_overlap.py nwchem.out
"""

import numpy as np
import sys
import re


def parse_overlap_matrix(filename):
    """
    Parse overlap matrix from custom NWChem output format
    
    Args:
        filename: NWChem output file
        
    Returns:
        S: overlap matrix [nbf, nbf]
    """
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the overlap matrix section
    start_idx = None
    for i, line in enumerate(lines):
        if 'Overlap matrix' in line:
            start_idx = i
            break
    
    if start_idx is None:
        raise ValueError("Could not find 'Overlap matrix' in output file")
    
    print(f"Found overlap matrix at line {start_idx + 1}")
    
    # Skip the header line and dashes
    data_start = start_idx + 2
    
    # Parse all matrix elements
    elements = {}
    max_row = 0
    max_col = 0
    
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        
        # Stop at empty line or next section
        if not line or '=' in line or 'Matrix' in line or '---' in line:
            if elements:  # Only break if we've found some data
                break
        
        # Parse line: "1  2  0.6458989405E+00"
        parts = line.split()
        if len(parts) >= 3:
            try:
                row = int(parts[0]) - 1  # Convert to 0-indexed
                col = int(parts[1]) - 1
                value = float(parts[2])
                
                elements[(row, col)] = value
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                
            except (ValueError, IndexError):
                # Not a data line, skip
                continue
    
    if not elements:
        raise ValueError("No matrix elements found")
    
    # Determine matrix size
    nbf = max(max_row, max_col) + 1
    print(f"Detected matrix size: {nbf} x {nbf}")
    print(f"Found {len(elements)} matrix elements")
    
    # Construct matrix
    S = np.zeros((nbf, nbf))
    
    for (row, col), value in elements.items():
        S[row, col] = value
    
    return S


def verify_overlap(S):
    """
    Verify overlap matrix properties
    """
    
    print("\n" + "="*70)
    print("OVERLAP MATRIX VERIFICATION")
    print("="*70)
    
    # Display matrix
    print(f"\nOverlap matrix S ({S.shape[0]}x{S.shape[1]}):")
    print(S)
    
    # Check symmetry
    is_symmetric = np.allclose(S, S.T, atol=1e-10)
    asymmetry = np.max(np.abs(S - S.T))
    
    print(f"\n✓ Symmetry:")
    print(f"  Symmetric: {is_symmetric}")
    print(f"  Max asymmetry: {asymmetry:.2e}")
    
    if not is_symmetric and asymmetry > 1e-8:
        print(f"  ⚠ Warning: Matrix not perfectly symmetric")
        print(f"  Symmetrizing...")
        S = (S + S.T) / 2
    
    # Check diagonal
    diag = np.diag(S)
    print(f"\n✓ Diagonal:")
    print(f"  Values: {diag}")
    print(f"  All equal to 1.0: {np.allclose(diag, 1.0)}")
    
    if not np.allclose(diag, 1.0):
        print(f"  ⚠ Warning: Diagonal not all 1.0")
        print(f"  Range: [{diag.min():.8f}, {diag.max():.8f}]")
    
    # Check eigenvalues (positive definite)
    eigvals = np.linalg.eigvalsh(S)
    print(f"\n✓ Eigenvalues:")
    print(f"  Min: {eigvals.min():.8f}")
    print(f"  Max: {eigvals.max():.8f}")
    print(f"  All positive: {np.all(eigvals > 0)}")
    
    if eigvals.min() <= 0:
        print(f"  ⚠ Warning: Matrix not positive definite!")
    
    # Trace
    trace = np.trace(S)
    print(f"\n✓ Trace: {trace:.8f}")
    print(f"  Expected: {S.shape[0]} (number of basis functions)")
    
    # Off-diagonal elements
    mask = ~np.eye(S.shape[0], dtype=bool)
    offdiag = S[mask]
    print(f"\n✓ Off-diagonal elements:")
    print(f"  Count: {len(offdiag)}")
    print(f"  Range: [{offdiag.min():.8f}, {offdiag.max():.8f}]")
    print(f"  Mean: {offdiag.mean():.8f}")
    
    print("\n" + "="*70)
    
    return S


def verify_with_density(S, density_file='h2plus_training_data.npz'):
    """
    Verify Tr(ρS) = N_electrons
    
    Args:
        S: overlap matrix
        density_file: NPZ file with density matrices
    """
    
    try:
        data = np.load(density_file)
    except FileNotFoundError:
        print(f"\n⚠ Density file not found: {density_file}")
        print("  Skipping density verification")
        return
    
    density_real = data['density_real']
    density_imag = data['density_imag']
    
    print("\n" + "="*70)
    print("DENSITY VERIFICATION")
    print("="*70)
    
    # Check several timesteps
    timesteps = [0, len(density_real)//2, -1]
    
    for i, idx in enumerate(timesteps):
        rho = density_real[idx] + 1j * density_imag[idx]
        
        # Compute Tr(ρ)
        tr_rho = np.trace(rho).real
        
        # Compute Tr(ρS)
        rho_S = rho @ S
        tr_rho_S = np.trace(rho_S).real
        
        print(f"\nTimestep {idx} (index {idx}):")
        print(f"  Tr(ρ)  = {tr_rho:.10f}")
        print(f"  Tr(ρS) = {tr_rho_S:.10f}")
        
        # Check if Tr(ρS) ≈ 1.0 for H2+
        if np.abs(tr_rho_S - 1.0) < 0.01:
            print(f"  ✓✓✓ CORRECT! Tr(ρS) ≈ 1.0 (H2+ has 1 electron)")
        else:
            print(f"  ⚠ Warning: Tr(ρS) ≠ 1.0")
            print(f"  Expected 1.0 for H2+ (1 electron)")
    
    # Check conservation
    traces = [np.trace((density_real[i] + 1j * density_imag[i]) @ S).real 
              for i in range(len(density_real))]
    
    print(f"\n✓ Conservation over time:")
    print(f"  Mean Tr(ρS): {np.mean(traces):.10f}")
    print(f"  Std Tr(ρS):  {np.std(traces):.2e}")
    print(f"  Min Tr(ρS):  {np.min(traces):.10f}")
    print(f"  Max Tr(ρS):  {np.max(traces):.10f}")
    
    if np.std(traces) < 0.001:
        print(f"  ✓✓✓ Excellently conserved!")
    
    print("\n" + "="*70)


def main():
    
    if len(sys.argv) < 2:
        print("Usage: python parse_custom_overlap.py nwchem.out")
        print("\nParses overlap matrix from modified NWChem output")
        print("\nExpected format:")
        print("  Overlap matrix")
        print("  --------------")
        print("      1     1    0.1000000000E+01")
        print("      1     2    0.6458989405E+00")
        print("      ...")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    print("="*70)
    print("PARSING OVERLAP MATRIX FROM NWCHEM OUTPUT")
    print("="*70)
    print(f"\nInput file: {output_file}\n")
    
    # Parse
    try:
        S = parse_overlap_matrix(output_file)
    except Exception as e:
        print(f"\n✗ Error parsing overlap matrix: {e}")
        sys.exit(1)
    
    print(f"\n✓ Successfully parsed overlap matrix")
    
    # Verify properties
    S = verify_overlap(S)
    
    # Save to files
    base_name = output_file.replace('.out', '')
    
    # NumPy binary format
    npy_file = f"{base_name}_overlap.npy"
    np.save(npy_file, S)
    print(f"\n✓ Saved binary: {npy_file}")
    
    # Text format (for inspection)
    txt_file = f"{base_name}_overlap.txt"
    np.savetxt(txt_file, S, fmt='%16.10f', 
               header=f"Overlap matrix S ({S.shape[0]}x{S.shape[1]})")
    print(f"✓ Saved text: {txt_file}")
    
    # Python code to load (for convenience)
    py_file = f"{base_name}_overlap.py"
    with open(py_file, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write("# Overlap matrix from NWChem\n")
        f.write(f"# Source: {output_file}\n")
        f.write(f"# Size: {S.shape[0]}x{S.shape[1]}\n\n")
        f.write("S = np.array([\n")
        for row in S:
            f.write("    [" + ", ".join(f"{x:.10f}" for x in row) + "],\n")
        f.write("])\n")
    print(f"✓ Saved Python: {py_file}")
    
    # Verify with density if available
    verify_with_density(S)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Overlap matrix successfully extracted")
    print(f"  Size: {S.shape[0]}x{S.shape[1]}")
    print(f"  Trace: {np.trace(S):.8f}")
    print(f"  Files created:")
    print(f"    - {npy_file} (load with np.load)")
    print(f"    - {txt_file} (human-readable)")
    print(f"    - {py_file} (import directly)")
    
    print(f"\n✓ Next steps:")
    print(f"  1. Verify Tr(ρS) ≈ 1.0 (check DENSITY VERIFICATION above)")
    print(f"  2. Include S in your model (see overlap_matrix_inclusion.py)")
    print(f"  3. Train with geometry-aware features!")
    
    print("\n" + "="*70)
    print("✓✓✓ Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
