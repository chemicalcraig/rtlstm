import numpy as np
import glob
import os
import argparse

def get_time_from_restart(filepath):
    """
    Reads the simulation time 't' from an NWChem restart file.
    """
    with open(filepath, 'r') as f:
        # Read content (files are usually small enough to read fully)
        content = f.read().split()
        
    try:
        # Find the index of the 't' token and get the next value
        t_idx = content.index('t')
        time_val = float(content[t_idx + 1])
        return time_val
    except ValueError:
        print(f"Warning: Could not find time 't' in {filepath}")
        return None

def sync_field_data(density_dir, field_file, output_file, tolerance=1e-5):
    """
    Creates a new field file containing only entries that match the 
    times found in the density directory.
    """
    # 1. Load Field Data
    print(f"Loading field data from {field_file}...")
    try:
        # Assumes field.dat is text: time | Ex | Ey | Ez
        field_data = np.loadtxt(field_file) 
    except Exception as e:
        print(f"Error loading field file: {e}")
        return

    # 2. Get Density Times
    print(f"Scanning density files in '{density_dir}'...")
    # Matches any file with 'rt_restart' in the name
    density_files = glob.glob(os.path.join(density_dir, "*rt_restart*"))
    
    if not density_files:
        print("No density files found! Check the directory path.")
        return

    density_times = []
    for dp in density_files:
        t = get_time_from_restart(dp)
        if t is not None:
            density_times.append(t)
    
    # Sort times to ensure monotonic order
    density_times = np.sort(density_times)
    print(f"Found {len(density_times)} density snapshots.")
    print(f"Time range: {density_times[0]:.4f} to {density_times[-1]:.4f}")

    # 3. Filter Field Data
    # We maintain a list of matched rows
    matched_rows = []
    
    # We iterate through density times and find the closest field time
    # This is O(N*M) worst case, but efficient enough for typical datasets.
    # For very large matching, we could use searchsorted.
    
    field_times = field_data[:, 0]
    
    matched_count = 0
    
    for target_t in density_times:
        # Find index of field time closest to target_t
        # abs(field_times - target_t) gives diff array
        idx = (np.abs(field_times - target_t)).argmin()
        
        diff = abs(field_times[idx] - target_t)
        
        if diff < tolerance:
            matched_rows.append(field_data[idx])
            matched_count += 1
        else:
            print(f"Warning: No matching field found for density t={target_t:.6f} (closest diff={diff:.6e})")

    if matched_count == 0:
        print("Error: No timestamps matched! Check units (fs vs au).")
        return

    # 4. Save Output
    matched_array = np.array(matched_rows)
    
    # Save as .npy binary
    np.save(output_file, matched_array)
    
    # Optional: Save as .dat text for easy inspection or fallback
    txt_output = output_file.replace('.npy', '.dat')
    np.savetxt(txt_output, matched_array, header="time Ex Ey Ez")
    
    print("-" * 40)
    print(f"Synchronization Complete.")
    print(f"Original Field Steps: {len(field_data)}")
    print(f"Matched Field Steps:  {len(matched_array)}")
    print(f"Saved binary to:      {output_file}")
    print(f"Saved text to:        {txt_output}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync field.dat with density restart files.")
    parser.add_argument("--densities", type=str, default="densities", help="Directory containing restart files")
    parser.add_argument("--field", type=str, default="field.dat", help="Path to original field.dat")
    parser.add_argument("--out", type=str, default="field_modified.npy", help="Output .npy file")
    
    args = parser.parse_args()
    
    sync_field_data(args.densities, args.field, args.out)
