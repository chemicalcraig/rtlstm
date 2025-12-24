# RTLSTM: Real-Time Quantum Dynamics with LSTMs

## Project Overview

RTLSTM is a machine learning framework designed to accelerate Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) simulations. It uses an LSTM architecture with residual connections to predict the time-evolution of the density matrix, bypassing expensive Kohn-Sham integration steps.

The project features two main approaches:

1.  **LSTM-based Model**: A `DensityMatrixLSTM` model that predicts the density matrix for the next time step based on a sequence of previous density matrices and the external electric field. This model can use either a first-order Euler or a more sophisticated second-order Verlet integration scheme.

2.  **Graph-based Model**: A `DensityGraphNet` model that represents the density matrix as a graph, where the basis functions are the nodes and the overlap matrix defines the edges. This approach is more advanced and allows for greater transferability between different molecular systems.

Both models are physics-informed, incorporating constraints such as Hermiticity and trace preservation to ensure physically meaningful predictions. The project also utilizes a "skip-step" training curriculum to improve long-term stability and a "rollout" training scheme to minimize autoregressive drift.

The project is written in Python and uses the following main technologies:

*   **PyTorch**: For building and training the deep learning models.
*   **torch-geometric**: For the graph-based model.
*   **NumPy**: For numerical operations.
*   **Matplotlib**: For plotting and visualization.

## Building and Running

### 1. Installation

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

The project expects data from a quantum chemistry calculation (e.g., from NWChem). The following scripts are provided for pre-processing:

*   **`src/rtparse.py`**: Parses an NWChem output file to extract various time-series data, such as the dipole moment, electric field, and molecular orbital occupations.
*   **`src/get_overlap.py`**: Extracts the overlap matrix from an NWChem output file.

Example usage:

```bash
# Create directories for the data
mkdir densities rt_data data perm

# Parse the NWChem output file
python src/rtparse.py NWCALC.out
mv *.dat rt_data/

# Extract the overlap matrix
python src/get_overlap.py NWCALC.out
mv *overlap.npy data/
mv *overlap.txt data/
```

### 3. Training

The training process is configured via a JSON file. The main training scripts are:

*   **`src/train_benchmark.py`**: For the LSTM-based model.
*   **`src/train_graph_skip.py`**: For the graph-based model with skip-step training.

Example usage:

```bash
# Synchronize the field data
python src/sync_field.py --densities perm/ --field rt_data/field.dat --out data/field.npy

# Aggregate the density data
python src/aggregate_data.py --dir perm/ --out densities/density_series.npy

# Train the model
python src/train_benchmark.py --config inputs/train.json
```

### 4. Prediction

Predictions are also configured via a JSON file. The main prediction script is `src/predict_config.py`.

Example usage:

```bash
# Extract the initial densities for bootstrapping
python src/extract_densities.py --n-densities 20 CALCNAME.rt_restart

# Run the prediction
python src/predict_config.py inputs/predict.json
```

### Quickstart

The project includes a `quickstart.py` script that demonstrates the full pipeline for the graph-based model.

```bash
# Run the full demo
python quickstart.py

# Run a faster demo with fewer epochs
python quickstart.py --quick

# Use a pretrained model (if available)
python quickstart.py --skip-training
```

## Development Conventions

*   **Configuration**: The training and prediction workflows are configured using JSON files.
*   **Physics-Informed**: The models incorporate physical constraints to ensure the predictions are physically meaningful.
*   **Modularity**: The code is organized into modules for data loading, model definition, training, and prediction.
*   **Pre-processing**: The project relies on a pre-processing step to extract and format data from quantum chemistry software outputs.
*   **Visualization**: The `quickstart.py` script provides an example of how to visualize the results using Matplotlib.
