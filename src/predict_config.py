import torch
import numpy as np
import json
import sys
import os
from train_benchmark import DensityMatrixLSTM, CFG # Import class definition

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

	# Load Model
    model = DensityMatrixLSTM().to(CFG['device'])
    model.load_state_dict(torch.load(config['io']['model_path'], map_location=CFG['device']))
    model.eval()
    
    # --- 1. Load Initial History ---
    init_cfg = config['initial_state']
    mode = init_cfg.get('mode', 'groundstate')
    
    # This tensor will hold the history we want to SAVE to the file
    initial_history_tensor = None 
    consumed_steps = 0

    if mode == 'warmstart':
        file_list = init_cfg['bootstrap_files']
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
    # (Same field loading logic as before)
    field_cfg = config['field']
    if field_cfg.get('type') == 'file':
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
