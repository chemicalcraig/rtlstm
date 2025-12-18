import numpy as np
import glob
import os
import argparse

def get_time_from_restart(filepath):
    """
    Reads the simulation time 't' from an NWChem restart file.
    """
    with open(filepath, 'r') as f:
        content = f.read().split()
        
    try:
        t_idx = content.index('t')
        time_val = float(content[t_idx + 1])
        return time_val
    except ValueError:
        print(f"Warning: Could not find time 't' in {filepath}")
        return None

def sync_field_data(density_dir, field_file, output_file, tolerance=1e-5):
    """
    Creates a new field file matched to density times.
    If field_file is empty, generates zero-field entries.
    """
    # 1. Load Field Data
    print(f"Loading field data from {field_file}...")
    field_data = np.array([])
    try:
        # Load, treating '#' as comments. If file is only comments, returns empty.
        field_data = np.loadtxt(field_file) 
    except Exception as e:
        print(f"Note: Error reading field file or file empty ({e}). Proceeding with zeros.")
        field_data = np.array([])

    # 2. Get Density Times
    print(f"Scanning density files in '{density_dir}'...")
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
    if len(density_times) > 0:
        print(f"Time range: {density_times[0]:.4f} to {density_times[-1]:.4f}")

    # 3. Filter or Generate Field Data
    matched_rows = []

    # CHECK: Is the field data empty?
    if field_data.size == 0:
        print(">>> No external field data found. Generating ZERO field for all steps.")
        for t in density_times:
            # Create a row: [time, Ex, Ey, Ez] -> [t, 0.0, 0.0, 0.0]
            matched_rows.append([t, 0.0, 0.0, 0.0])
            
    else:
        # Standard matching logic
        field_times = field_data[:, 0]
        matched_count = 0
        
        for target_t in density_times:
            # Find closest time index
            idx = (np.abs(field_times - target_t)).argmin()
            diff = abs(field_times[idx] - target_t)
            
            if diff < tolerance:
                matched_rows.append(field_data[idx])
                matched_count += 1
            else:
                print(f"Warning: No matching field found for density t={target_t:.6f} (closest diff={diff:.6e})")

        if matched_count == 0:
            print("Error: Field file has data, but NO timestamps matched! Check units (fs vs au).")
            return

    # 4. Save Output
    matched_array = np.array(matched_rows)
    
    if len(matched_array) == 0:
        print("Error: Resulting array is empty. Aborting save.")
        return

    # Save as .npy binary
    np.save(output_file, matched_array)
    
    # Optional: Save as .dat text for inspection
    txt_output = output_file.replace('.npy', '.dat')
    np.savetxt(txt_output, matched_array, header="time Ex Ey Ez")
    
    print("-" * 40)
    print(f"Synchronization Complete.")
    if field_data.size > 0:
        print(f"Original Field Steps: {len(field_data)}")
    else:
        print(f"Original Field Steps: 0 (Auto-filled zeros)")
        
    print(f"Matched Field Steps:  {len(matched_array)}")
    print(f"Saved binary to:      {output_file}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync field.dat with density restart files.")
    parser.add_argument("--densities", type=str, default="densities", help="Directory containing restart files")
    parser.add_argument("--field", type=str, default="field.dat", help="Path to original field.dat")
    parser.add_argument("--out", type=str, default="field_modified.npy", help="Output .npy file")
    
    args = parser.parse_args()
    
    sync_field_data(args.densities, args.field, args.out)