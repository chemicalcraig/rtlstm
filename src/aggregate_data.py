import torch
import numpy as np
import glob
import os
import re
import argparse

def parse_nwchem_restart(filepath):
    """
    Parses a single NWChem RT-TDDFT restart file for Open-Shell systems.
    
    Structure:
    - nmats: Number of spin matrices (2 for Open Shell: Alpha, Beta)
    - Data stream: Interleaved Real/Imaginary values
      (Real, Imag, Real, Imag, ...)
    """
    with open(filepath, 'r') as f:
        content = f.read().split()

    try:
        # --- 1. Extract Metadata ---
        nbf_idx = content.index('nbf_ao')
        n_basis = int(content[nbf_idx + 1])
        
        nmats_idx = content.index('nmats')
        n_mats = int(content[nmats_idx + 1]) # Expected: 2 (Alpha, Beta)
        
        t_idx = content.index('t')
        time = float(content[t_idx + 1])
        
        # --- 2. Extract Data Stream ---
        # Data starts after 'checksum' <value>
        checksum_idx = content.index('checksum')
        data_start_idx = checksum_idx + 2 
        
        # Load all remaining tokens as floats
        raw_data = np.array(content[data_start_idx:], dtype=np.float64)
        
    except ValueError as e:
        raise ValueError(f"Error parsing header in {filepath}: {e}")

    # --- 3. Validate Size ---
    # Each matrix element has 2 values (Real, Imag)
    # Total floats = n_mats * (n_basis * n_basis) * 2
    floats_per_matrix = (n_basis * n_basis) * 2
    expected_total = n_mats * floats_per_matrix
    
    if raw_data.size != expected_total:
        raise ValueError(
            f"Size mismatch in {filepath}.\n"
            f"Expected {expected_total} floats ({n_mats} mats x {n_basis}^2 elements x 2 parts)\n"
            f"Got {raw_data.size} floats."
        )

    # --- 4. Process Matrices (Alpha & Beta) ---
    spins_list = []
    
    # Iterate over the number of matrices (spins)
    for m in range(n_mats):
        # Slice the data for this specific matrix
        start = m * floats_per_matrix
        end = start + floats_per_matrix
        matrix_data = raw_data[start:end]
        
        # Apply the Interleaved Rule:
        # Even indices (0, 2, 4...) -> Real
        # Odd indices (1, 3, 5...) -> Imag
        real_part = matrix_data[0::2]
        imag_part = matrix_data[1::2]
        
        # Reshape to (N, N)
        rho_real = torch.tensor(real_part.reshape(n_basis, n_basis), dtype=torch.float64)
        rho_imag = torch.tensor(imag_part.reshape(n_basis, n_basis), dtype=torch.float64)
        
        # Combine into Complex Tensor
        rho_complex = torch.complex(rho_real, rho_imag)
        spins_list.append(rho_complex)

    # Stack Spins: Shape (2, N, N) -> [Alpha, Beta]
    rho_combined = torch.stack(spins_list)
    
    return time, rho_combined

def aggregate_restart_files(directory, output_file="density_series.npy"):
    """
    Aggregates restart files into a single tensor: (Time, Spins, N, N)
    """
    # Filter for files ending in numeric digits
    file_pattern = os.path.join(directory, "*rt_restart*")
    files = glob.glob(file_pattern)
    files = [f for f in files if re.search(r'\.\d+$', f)]
    files.sort(key=lambda x: int(re.search(r'\.(\d+)$', x).group(1)))
    
    if not files:
        print(f"No restart files found in {directory}")
        return

    print(f"Found {len(files)} restart files. Aggregating Open-Shell data...")

    rho_list = []
    times = []
    
    for i, fp in enumerate(files):
        try:
            t, rho = parse_nwchem_restart(fp) # rho is (2, N, N)
            rho_list.append(rho)
            times.append(t)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(files)}...")
                
        except Exception as e:
            print(f"Failed to parse {fp}: {e}")
            break

    if rho_list:
        # Stack time steps
        # Final Shape: (Time, Spins, N, N)
        rho_series = torch.stack(rho_list)
        time_series = np.array(times)
        
        save_dict = {
            "density": rho_series.numpy(),
            "time": time_series
        }
        
        np.save(output_file, save_dict)
        print(f"\nSuccess! Saved to: {output_file}")
        print(f"Shape: {rho_series.shape} (Time, Spins, N_basis, N_basis)")
        
        # Quick validation print
        if rho_series.shape[1] == 2:
            print("Verified: Detected 2 spin channels (Alpha/Beta).")
    else:
        print("Aggregation failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing restart files")
    parser.add_argument("--out", type=str, default="density_series.npy", help="Output filename")
    args = parser.parse_args()
    
    aggregate_restart_files(args.dir, args.out)
