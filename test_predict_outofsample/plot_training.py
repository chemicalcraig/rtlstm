import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

plot_steps = 125
row_to_plot = 0
outfile = "predicted_vs_sim_"+str(row_to_plot)+"x_imag.png"

def plot_diagonal_evolution(pred_file, gt_file="densities/density_series.npy", save_path=outfile):
    # --- 1. Load Prediction Data ---
    if not os.path.exists(pred_file):
        print(f"Error: Prediction file '{pred_file}' not found.")
        return
    pred_data = np.load(pred_file)
    print(f"Loaded Prediction shape: {pred_data.shape}")

    # --- 2. Load Ground Truth Data ---
    gt_data = None
    if os.path.exists(gt_file):
        try:
            # density_series.npy is saved as a dictionary {'density': ..., 'time': ...}
            gt_dict = np.load(gt_file, allow_pickle=True).item()
            gt_data = gt_dict['density']
            print(f"Loaded Ground Truth shape: {gt_data.shape}")
        except Exception as e:
            print(f"Warning: Could not load ground truth from {gt_file}. Error: {e}")
    else:
        print(f"Warning: Ground truth file '{gt_file}' not found. Plotting prediction only.")

    # --- 3. Setup Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Settings
    spin_idx = 0  # Index 0 = Alpha spin
    
    # --- 4. Plotting Loop ---
    for i in range(4):
        ax = axes[i]
        
        # --- Plot Ground Truth (Background) ---
        if gt_data is not None:
            # Ensure we don't go out of bounds if GT is shorter than plot_steps
            limit = min(plot_steps, len(gt_data))
            try:
                # Ground Truth style: Black solid line
                trace_gt = gt_data[:limit, spin_idx, row_to_plot, i].imag
                ax.plot(np.arange(limit), trace_gt, color='k', linewidth=1.5, alpha=0.6, label='Ground Truth')
            except IndexError:
                # Fallback for shape mismatch
                pass

        # --- Plot Prediction (Foreground) ---
        # Ensure we don't go out of bounds
        limit = min(plot_steps, len(pred_data))
        try:
            # Prediction style: Blue line with dots
            trace_pred = pred_data[:limit, spin_idx, row_to_plot, i].imag
            ax.plot(np.arange(limit), trace_pred, linewidth=2, label=f'Pred ρ[{row_to_plot},{i}]', marker=".", markersize=4)
        except IndexError:
            trace_pred = pred_data[:limit, row_to_plot, i].imag
            ax.plot(np.arange(limit), trace_pred, linewidth=2, label=f'Pred ρ[{row_to_plot},{i}]', marker=".")

        # --- Formatting ---
        ax.set_title(f"Off-Diagonal Element ({row_to_plot},{i})", fontsize=12)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(f"ρ[{row_to_plot},{i}] (imag)", fontsize=11)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.6)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # You can change the paths here if needed
    plot_diagonal_evolution("predicted_dynamics.npy", "densities/density_series.npy")



#import matplotlib.pyplot as plt
#import numpy as np
#
#temp = np.load("h2_plus_rttddft_training_data.npz")
#np.save('first_10_densities.npy', temp['density_real'][:10])
#print(temp['density_real'][:10])
#
#n_electrons = np.trace(temp['density_real'][0] + 1j * temp['density_imag'][0]).real
#print(n_electrons)
#predictions = temp['density_real'][:4000]
#print("\n1. Plotting density matrix elements...")
#fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#axes = axes.flatten()
#n_steps = len(predictions)
#n_basis = predictions.shape[1]
#timesteps = np.arange(n_steps)
#
#for i in range(min(4, n_basis)):
#    ax = axes[i]
#    rho_diag = np.array([predictions[t, i, i].real for t in range(4000)])
#    
#    ax.plot(timesteps, rho_diag, linewidth=2, label=f'ρ[{i},{i}]')
#    ax.set_xlabel('Step', fontsize=11)
#    ax.set_ylabel(f'ρ[{i},{i}] (real)', fontsize=11)
#    ax.set_title(f'Diagonal Element ({i},{i})', fontsize=12)
#    ax.legend(fontsize=9)
#    ax.grid(True, alpha=0.3)
#
#plt.tight_layout()
#plt.savefig(f'training_density_evolution.png', dpi=300, bbox_inches='tight')
#print(f"   Saved: training_density_evolution.png")
#plt.close()
#
#
