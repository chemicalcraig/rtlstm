import torch
import numpy as np
import json
import sys
import os
from pathlib import Path
from train_benchmark import DensityMatrixLSTM, EnsembleDensityMatrixLSTM, CFG # Import class definitions
from timestep_sync import (
    get_model_dt,
    parse_field_timestamps,
    sync_bootstrap_to_model,
    validate_timesteps
)

def load_single_rho(filepath, n_basis, dtype=torch.complex128):
    arr = np.load(filepath)
    tensor = torch.tensor(arr, dtype=dtype)
    
    # --- CONSISTENCY FIX ---
    # If training data was conjugated to fix time-reversal,
    # we must conjugate inputs here too.
#    v = tensor.diag()
#    mask = torch.diag(torch.ones_like(tensor))
#    mask = mask.conj()
#    out = mask*torch.diag(tensor) + (1. - mask)*tensor
    tensor = tensor.conj()
    
    if tensor.ndim == 2: return torch.stack([tensor, torch.zeros_like(tensor)])
    return tensor

def generate_pulse_field(steps, dt, amplitude, freq):
    t = torch.linspace(0, steps*dt, steps)
    envelope = torch.exp(-(t - (steps*dt)/2)**2 / 100) 
    field_val = amplitude * torch.sin(freq * t) * envelope
    field = torch.zeros((steps, 3), dtype=torch.float64)
    field[:, 0] = field_val 
    return field


def run_prediction(config_path):
    with open(config_path, 'r') as f: config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running prediction on {device}...")

    # Setup Global CFG for Model Init
    CFG['n_basis'] = config['model']['n_basis']
    CFG['hidden_dim'] = config['model']['hidden_dim']
    CFG['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG['spin-polarization'] = config['system']['spin-polarization']
    seq_len = config['model']['seq_len']
    n_basis = CFG['n_basis']

    # --- Timestep Validation ---
    model_path = config['io']['model_path']
    model_dt = get_model_dt(model_path)

    # If model doesn't have stored dt, try to get from config
    if model_dt is None:
        model_dt = config.get('field', {}).get('dt', None)
        if model_dt is None:
            print("WARNING: Could not determine model training dt. Skipping timestep validation.")
        else:
            print(f"Using dt={model_dt} from prediction config (model has no stored dt)")
    else:
        print(f"Model training dt: {model_dt:.6f}")

    # Load Model (handle both old state_dict and new checkpoint format)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Check if this is an ensemble model
    is_ensemble = False
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        is_ensemble = checkpoint['config'].get('use_ensemble', False)

    if is_ensemble:
        # Load masks from checkpoint
        masks = checkpoint.get('masks', None)
        if masks is None:
            raise ValueError("Ensemble model checkpoint missing 'masks' key")

        # Update CFG with ensemble params from checkpoint
        ckpt_cfg = checkpoint['config']
        CFG['use_verlet'] = ckpt_cfg.get('use_verlet', False)
        CFG['verlet_damping'] = ckpt_cfg.get('verlet_damping', 0.1)
        CFG['verlet_blend'] = ckpt_cfg.get('verlet_blend', 0.5)
        CFG['trace_projection'] = ckpt_cfg.get('trace_projection', True)
        CFG['n_alpha'] = ckpt_cfg.get('n_alpha', 1.0)
        CFG['n_beta'] = ckpt_cfg.get('n_beta', 0.0)
        CFG['overlap_file'] = config['model'].get('overlap_file', '')

        model = EnsembleDensityMatrixLSTM(masks).to(device)
        print(f"Loaded ensemble model with {len(masks)} sub-models")
    else:
        model = DensityMatrixLSTM().to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # --- 1. Load Initial History ---
    init_cfg = config['initial_state']
    mode = init_cfg.get('mode', 'groundstate')

    # This tensor will hold the history we want to SAVE to the file
    initial_history_tensor = None
    consumed_steps = 0
    synced_field = None  # Will hold field values if resampled

    if mode == 'warmstart':
        file_list = init_cfg['bootstrap_files']
        field_cfg = config['field']

        # --- Timestep Sync Check ---
        if model_dt is not None and field_cfg.get('type') == 'file':
            field_path = field_cfg['path']

            # Infer density directory from bootstrap files
            if file_list:
                density_dir = str(Path(file_list[0]).parent)
            else:
                density_dir = None

            print(f"Checking timestep compatibility...")
            sync_result = sync_bootstrap_to_model(
                bootstrap_files=file_list,
                field_path=field_path,
                model_dt=model_dt,
                density_dir=density_dir,
                tolerance=0.05  # 5% tolerance
            )

            print(f"  {sync_result['message']}")

            if sync_result['was_resampled']:
                file_list = sync_result['synced_files']
                synced_field = torch.tensor(sync_result['synced_field'], dtype=torch.float64)
                print(f"  Resampled to {len(file_list)} density files")

        print(f"Mode: Warm-Start. Loading first {seq_len} of {len(file_list)} files...")

        # Load necessary history for model input
        files_to_load = file_list[:seq_len]
        consumed_steps = seq_len

        history_list = []
        for fp in files_to_load:
            rho = load_single_rho(fp, n_basis)
            history_list.append(rho)

        # (Batch=1, Seq, Spin, N, N)
        current_seq = torch.stack(history_list).unsqueeze(0).to(device)

        # Save this history for the output file
        initial_history_tensor = torch.stack(history_list) # (Seq, Spin, N, N)

    else:
        # Cold Start
        print("Mode: Cold-Start.")
        gs_file = init_cfg['ground_state_file']
        rho_gs = load_single_rho(gs_file, n_basis)
        current_seq = rho_gs.unsqueeze(0).unsqueeze(1).repeat(1, seq_len, 1, 1, 1).to(device)
        consumed_steps = 0

        # For cold start, history is just repeated GS
        initial_history_tensor = rho_gs.unsqueeze(0).repeat(seq_len, 1, 1, 1)

    # --- 2. Prepare Field ---
    # Use synced field if resampling was performed, otherwise load from file
    if synced_field is not None:
        # Already resampled to match model dt
        field_series = synced_field
        print(f"Using resampled field ({len(field_series)} steps)")
    elif field_cfg.get('type') == 'file':
        field_arr = np.loadtxt(field_cfg['path'])
        if field_arr.ndim == 1: field_arr = field_arr[1:] if len(field_arr)==4 else field_arr
        elif field_arr.shape[1] == 4: field_arr = field_arr[:, 1:]
        field_series = torch.tensor(field_arr, dtype=torch.float64)
    else:
        field_series = generate_pulse_field(field_cfg['steps'], field_cfg['dt'], field_cfg['amplitude'], field_cfg['frequency'])

    if consumed_steps > 0:
        field_series = field_series[consumed_steps:]

    field_series = field_series.to(device)

    # --- 3. Run Prediction ---
    predictions = []
    total_steps = len(field_series)
    print(f"Predicting {total_steps} steps...")

    with torch.no_grad():
        for t in range(total_steps):
            next_field = field_series[t].unsqueeze(0) # (1, 1, 3)
            
            # Predict
            pred_rho = model(current_seq, next_field)
            #pred_rho = pred_rho.conj()
            predictions.append(pred_rho.squeeze(0).cpu()) # Store (Spin, N, N)
            
            # Update History
            current_seq = torch.cat([current_seq[:, 1:], pred_rho.unsqueeze(1)], dim=1)

    # --- 4. Stitch and Save ---
    # Combine: [Initial History] + [Predictions]
    predicted_tensor = torch.stack(predictions) # (T_pred, Spin, N, N)
    full_trajectory = torch.cat([initial_history_tensor, predicted_tensor], dim=0) # (Total, Spin, N, N)
    
    output_path = config['io']['output_file']
    np.save(output_path, full_trajectory.numpy())
    
    print(f"Done. Full trajectory (History + Prediction) saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_config.py prediction.json")
    else:
        run_prediction(sys.argv[1])
