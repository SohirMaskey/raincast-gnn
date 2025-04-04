# GNN Post-Processing for Precipitation Forecasting

This repository trains and evaluates a graph-based model for precipitation forecasting using plain PyTorch + PyTorch Geometric. Below are instructions for installing dependencies, training single runs, evaluating them, and running multiple experiments via a bash script.

---

## Table of Contents

1. [Installation](#installation)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Single-Run Training](#single-run-training)
5. [Single-Run Evaluation](#single-run-evaluation)
6. [Running All Experiments via Bash](#running-all-experiments-via-bash)
7. [Where to Find Results](#where-to-find-results)
8. [License](#license)

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/username/gnn-postprocessing.git
   cd gnn-postprocessing

2. **Install** conda.

3. **Create** environment.

    ```bash
    conda env create -f environment.yml
    conda activate gnn-env

## Repository Structure

gnn-postprocessing/
├── environment.yml             # Conda environment definition
├── train.py                    # Plain PyTorch training script
├── eval.py                     # Plain PyTorch evaluation script
├── models/
│   └── gnn.py                  # GNN model definition
├── utils/
│   ├── dataset.py              # EUPPBench or custom dataset code
│   └── data.py                 # Possibly code for building graphs
├── trained_models/
│   └── 24h_mixed_u/            # Example directory for storing logs, checkpoints, etc.
├── run_all_train.sh            # Example bash script to run multiple training
└── run_all_eval.sh             # Example bash script to run multiple evaluation

## Environment Setup

The environment.yml file defines a conda environment named gnn-env with:

    - Python 3.9

    - PyTorch 2.0.1 (compatible with CUDA 11.8)

    - PyTorch Geometric (2.3.1+)

    - Other libraries: NumPy, pandas, scikit-learn, geopy, etc.

Install and activate:
    ```bash
    conda env create -f environment.yml
    conda activate gnn-env

# Single-Run Training

You can do a single run (for example, 24-hour lead time, mixed config) by specifying:
    ```bash
    python train.py \
    --leadtime 24h \
    --dir trained_models/24h_mixed_u \
    --run_id 0

What happens:

    - train.py looks for params.json in trained_models/24h_mixed_u/ (make sure it exists).

    - Creates logs in trained_models/24h_mixed_u/train_0.log.

    - Saves the best checkpoint to trained_models/24h_mixed_u/models/run_0-best.ckpt.

# Single-Run Evaluation

Similarly, evaluate with:
    ```bash
    python eval.py \
    --leadtime 24h \
    --folder trained_models/24h_mixed_u \
    --data f

What happens:

    - eval.py looks for params.json in trained_models/24h_mixed_u/.

    - Finds .ckpt files in trained_models/24h_mixed_u/models/.

    - Averages predictions across them (if multiple exist).

    - Logs CRPS & saves eval_f.log, plus a CSV in trained_models/24h_mixed_u/f_results.csv (or f_results.txt summary).