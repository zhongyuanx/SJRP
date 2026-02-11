# SJRP: Deep Learning for Stochastic Joint Replenishment

This repository implements a deep learning approach to solving the stochastic joint replenishment problem (SJRP) using impulse control theory. It complements the Julia codes for the paper "A Computational Method for Solving the Stochastic Joint Replenishment Problem in High Dimensions" available at https://arxiv.org/pdf/2511.11830. It includes neural network training for value function approximation, policy extraction via diffusion and intervention operators, Monte Carlo performance simulation, and classical benchmark policies (R,S), (Q,S), and can-order. Two example configurations are provided here, a 1-dimensional configuration and a 12-dimensional one. 

## Pipeline Overview

The workflow has four stages:

1. **Benchmark computation** -- Compute (R,S), (Q,S), and can-order benchmark policies. The benchmark outputs also provide hyperparameters (ordering frequency, order-up-to vector) used by the training stage. The base config uses (R,S)/(Q,S) outputs; the can-order configs use can-order outputs.

2. **Training** -- Train neural networks (Vnet for value function, Znet for its gradient) by minimizing a penalized martingale loss over simulated sample paths.

3. **Policy extraction and simulation** -- Load trained models, compute the diffusion operator to determine when to order, solve for the optimal order-up-to level via LBFGS, and run Monte Carlo simulation to estimate long-run discounted cost.

4. **Diagnostics** -- Visualize the objective landscape and inspect trained model behavior.

5. **1D verification** -- `1d_solver.py` is a self-contained 1D solver (imports only `networks.py`) that can be used to verify the training pipeline on a simple scalar problem with hardcoded parameters.

## File Structure

```
.
├── main.py                               # Training entry point (CLI)
├── impulse_control_solver.py             # Deep learning solver (training loop)
├── networks.py                           # Vnet / Znet neural network definitions
├── nn_simulation.py                      # Policy extraction + performance simulation
├── benchmarks.py                         # (R,S), (Q,S), can-order benchmark policies
├── Sulem_sS_vectorized.py               # Analytical (s,S) policy solver (used by benchmarks)
├── nn_lbfgs_experiment.py               # LBFGS diagnostic tool with 1D plotting
├── generate_config.py                    # Generates 12-dim problem parameter CSVs
├── 1d_solver.py                      # Standalone 1D verification solver
│
├── configurations/
│   ├── 1dim/
│   │   ├── config.json                   # 1-dim problem configuration
│   │   ├── mu_1dim.csv, sigma_1dim.csv, ...
│   │   └── 1d_test/model_weights/        # Trained 1-dim model weights
│   └── 12dim/
│       ├── config.json                   # 12-dim problem configuration
│       ├── 12d_mhm_can_order_config.json       # Can-order training config (nu=0.2)
│       ├── 12d_mhm_can_order_nu4e-1_config.json  # Can-order training config (nu=0.4)
│       ├── mu_12dim.csv, sigma_12dim.csv, ...     # Problem parameter CSVs
│       ├── RS_R.csv, RS_S.csv, RS_cost.csv        # (R,S) benchmark outputs
│       ├── QS_Q.csv, QS_S.csv, QS_cost.csv        # (Q,S) benchmark outputs
│       ├── can_order_*.csv                         # Can-order benchmark outputs
│       ├── 12d_mhm_can_order/model_weights/       # Trained can-order weights (nu=0.2)
│       └── 12d_mhm_can_order_nu4e-1/model_weights/  # Trained can-order weights (nu=0.4)
│
└── 12d_mhm_model_weights/               # Trained 12-dim model weights
    ├── vnet_model.pth
    ├── znet_model.pth
    └── loss_history.csv
```

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- SciPy
- Munch
- Matplotlib (for diagnostics only)

## Usage

### 1. Compute benchmarks

```bash
# Run all benchmarks (RS, QS, can-order) for the 12-dim problem
python benchmarks.py --config configurations/12dim/config.json --run all

# Run only the (R,S) benchmark
python benchmarks.py --config configurations/12dim/config.json --run rs
```

### 2. Train neural networks

```bash
# 12-dim problem (base config, uses RS/QS reference process)
python main.py \
    --config_path configurations/12dim/config.json \
    --run_name 12d_mhm

# 12-dim problem (can-order reference process, nu=0.2)
python main.py \
    --config_path configurations/12dim/12d_mhm_can_order_config.json \
    --run_name 12d_mhm_can_order

# 1-dim problem
python main_with_path_resolution.py \
    --config_path configurations/1dim/config.json \
    --run_name 1d_test
```

Trained weights are saved to `<config_dir>/<run_name>/model_weights/`.

### 2b. 1D verification

```bash
# Standalone 1D solver with hardcoded parameters (no config file needed)
python 1d_solver.py
```

### 3. Evaluate trained policy

```bash
python nn_simulation.py \
    --config configurations/12dim/config.json \
    --model-weights 12d_mhm_model_weights \
    --diffop-eps -50.0 \
    --t-max 10000 \
    --num-samples 10000
```

This computes the optimal order-up-to level and simulates the discounted cost under the learned policy.

### 4. Diagnostics (1D)

```bash
# Plot the objective V(x) + c*x for the 1-dim model
python nn_lbfgs_experiment.py \
    --config configurations/1dim/config.json \
    --model-weights configurations/1dim/1d_test/model_weights \
    --plot --x-min 0 --x-max 10
```

## Configuration

Problem instances are specified via JSON config files. Each config has two sections:

- **`eqn_config`**: Problem parameters (dimension, discount rate, file paths for demand/cost data)
- **`net_config`**: Training hyperparameters (network architecture, batch size, iterations, learning rate / penalty schedules, reference process parameters)

See `configurations/12dim/config.json` for an example. Use `generate_config.py` to regenerate the 12-dim problem parameter CSVs.

Three 12-dim training configurations are provided:
- `config.json` -- uses (R,S)/(Q,S) benchmark outputs as reference process parameters
- `12d_mhm_can_order_config.json` -- uses can-order benchmark outputs, nu=0.2
- `12d_mhm_can_order_nu4e-1_config.json` -- uses can-order benchmark outputs, nu=0.4

Pre-trained model weights are included for all configurations so that evaluation can be run without retraining.
