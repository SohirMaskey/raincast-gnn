# GNN Post-Processing for Precipitation Forecasting

This repository trains and evaluates a graph-based model for precipitation forecasting using plain PyTorch + PyTorch Geometric. Below are instructions for installing dependencies, training single runs, evaluating them, and running multiple experiments via a bash script.

---

## Table of Contents

1. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Single-Run Training](#single-run-training)
5. [Single-Run Evaluation](#single-run-evaluation)
6. [Running All Experiments via Bash](#running-all-experiments-via-bash)
7. [Where to Find Results](#where-to-find-results)
8. [License](#license)

---

## Installation

1. **Clone** this repository:
   ```
   git clone https://github.com/username/gnn-postprocessing.git
   cd gnn-postprocessing
   ```
2. **Install** conda.

3. **Create** environment.
   ```
   conda env create -f environment.yml
   conda activate gnn-env
   ```
   
## Environment Setup

The environment.yml file defines a conda environment named gnn-env with:

    - Python 3.9

    - PyTorch 2.0.1 (compatible with CUDA 11.8)

    - PyTorch Geometric (2.3.1+)

    - Other libraries: NumPy, pandas, scikit-learn, geopy, etc.

Install and activate:
   ```
   conda env create -f environment.yml
   conda activate gnn-env
   ```
    

# Single-Run Training

You can do a single run (for example, 24-hour lead time, mixed config) by specifying:
    
    
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
    
    python eval.py \
    --leadtime 24h \
    --folder trained_models/24h_mixed_u \
    --data f
    
What happens:

- eval.py looks for params.json in trained_models/24h_mixed_u/.

- Finds .ckpt files in trained_models/24h_mixed_u/models/.

- Averages predictions across them (if multiple exist).
  
- Logs CRPS & saves eval_f.log, plus a CSV in trained_models/24h_mixed_u/f_results.csv (or f_results.txt summary).

# Running All Experiments via Bash

You can execute multiple runs incorporating

    leadtimes: 24h, 72h, 120h

    configs: normal, normal_mixed, mixed, mixed

For this run:
   ```
   chmod +x run_all_train.sh
   ./run_all_train.sh
   ```

To evaluate:
   ```
   chmod +x run_all_eval.sh
   ./run_all_eval.sh
   ```

   
# Where to Find Results

Each training subdirectory, e.g. trained_models/24h_mixed_u/, will contain:

    params.json: The hyperparameters.

    train_<run_id>.log: The training log.

    models/: The best checkpoint file(s).

    eval_<data>.log: The evaluation log.

    <data>_results.csv: CSV with final predictions.

    <data>.txt: CRPS summary.

# License


Happy Forecasting!
