# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTLSTM accelerates Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) simulations using LSTM neural networks. It predicts time-evolution of quantum density matrices to bypass expensive Kohn-Sham integration steps.

Two model architectures:
- **DensityMatrixLSTM** (`src/train_benchmark.py`): LSTM with residual learning, optional Verlet integration
- **DensityGraphNet** (`src/graph_model.py`): Graph neural network for transferable predictions across molecules

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Data Preparation (from NWChem output)
```bash
mkdir densities rt_data data perm
python src/rtparse.py NWCALC.out && mv *.dat rt_data/
python src/get_overlap.py NWCALC.out && mv *overlap.npy data/
python src/sync_field.py --densities perm/ --field rt_data/field.dat --out data/field.npy
python src/aggregate_data.py --dir perm/ --out densities/density_series.npy
```

### Training
```bash
# LSTM model
python src/train_benchmark.py --config inputs/train.json

# Graph model with skip-step curriculum
python src/train_graph_skip.py --config inputs/train_graph_skip.json
```

### Prediction
```bash
python src/extract_densities.py --n-densities 20 CALCNAME.rt_restart
python src/predict_config.py inputs/predict.json
```

### Running Tests
```bash
pytest test/
```

### Quickstart Demo
```bash
python quickstart.py           # Full demo
python quickstart.py --quick   # Fewer epochs
```

## Architecture

### Data Flow
```
NWChem .out → rtparse.py → .dat files
            → get_overlap.py → S.npy (overlap matrix)
            → extract_densities.py → bootstrap densities
            → sync_field.py + aggregate_data.py → training data
```

### Key Physics Constraints
- **Residual learning**: Predicts Δρ (change) not absolute ρ
- **Hermiticity**: Enforced via ρ = 0.5(ρ + ρ†)
- **Trace projection**: Conserves electron count via overlap matrix Tr(ρS) = N_electrons
- **Verlet integration**: Optional 2nd-order scheme for oscillatory dynamics

### Training Strategies
- **Rollout training**: Multi-step ahead prediction to minimize autoregressive drift
- **Skip-step curriculum** (`skip_step.py`): Gradually increase time-skip factor k (t → t + k·Δt)
- **Element weighting**: Inverse variance weighting for unbiased training

### Configuration
Training and prediction use JSON config files in `inputs/`. Key parameters in `train.json`:
- `model.seq_len`: LSTM history length (default: 20)
- `model.rollout_steps`: Multi-step prediction during training
- `model.use_verlet`: Enable Verlet integration for oscillatory systems
- `physics.lambda_trace/herm/idem`: Physics constraint weights

## Code Organization

- `src/` - Core modules (training, prediction, data processing)
- `inputs/` - JSON configuration templates
- `test/` - pytest unit tests
- `test_train/`, `test_predict_*` - Integration test directories with H₂⁺ examples
- `dashboard/` - Optional FastAPI backend + React frontend

## Development Notes

- Density matrices are complex-valued: shape `(2, n_basis, n_basis)` for open-shell (alpha/beta spin)
- Field data: external electric field vectors (3D per timestep)
- Always verify tensor shapes and complex number handling when modifying model code
- Test with `test_train/` example before modifying training pipeline
