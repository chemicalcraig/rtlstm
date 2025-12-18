RTLSTM: Real-Time Quantum Dynamics with LSTMs

RTLSTM is a machine learning framework designed to accelerate Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) simulations. It uses an LSTM architecture with residual connections to predict the time-evolution of the density matrix, bypassing expensive Kohn-Sham integration steps.

## Features
- **Residual Learning:** Predicts $\Delta \rho$ for stability.
- **Physics-Informed:** Constrains predictions via McWeeny purification and conservation laws.
- **Rollout Training:** Minimizes autoregressive drift during long-time inference.

## Installation
```bash
pip install -r requirements.txt
```
## Setup:
From working dir:
```bash
mkdir densities && mkdir rt_data && mkdir data && mkdir perm
python src/rtparse.py NWCALC.out && mv *.dat rt_data/
python src/get_overlap.py NWCALC.out && mv *overlap.npy data/ && mv *overlap.txt data/ && mv *overlap.txt data/
```

## To Train:
1. Copy train.json into working directory and modify accordingly
```bash
python src/sync_field.py --densities perm/ --field rt_data/field.dat --out data/field.npy
python src/aggregate_data.py --dir perm/ --out densities/density_series.npy
python src/train_benchmark.py --config inputs/train.json
```

## To Predict:
N.B. Must have a field file, if using, in data/field.dat. Extract ndensities for bootstrapping before prediction
1. Copy predict.json into working directory and modify accordingly
```bash
python src/extract_densities.py --n-densities 20 CALCNAME.rt_restart
python src/predict_config.py inputs/predict.json
```
