#!/usr/bin/env python3
"""
eval.py

Script to evaluate a trained GNN model in plain PyTorch (no PyTorch Lightning).
Logs to a file and console, includes seed setting for reproducibility, 
and manually ensembles multiple checkpoints if desired.
"""

import argparse
import json
import logging
import os
import sys
import random
import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader

# Example GNN
from models.gnn import GNN
# If we need custom logic
from utils.data import split_graph  # or remove if not needed
from utils.dataset import EUPPBench

############################################################################
# 1. Argparse
############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a GNN model (plain PyTorch).")
    parser.add_argument("--data", type=str, default="rf", choices=["rf","f"], help="Test set to use: 'rf' or 'f'.")
    parser.add_argument("--leadtime", type=str, default="24h", help="Lead time, e.g. '24h'.")
    parser.add_argument("--dir", type=str, required=True, help="Folder w/ 'params.json' & 'models/' subdir.")
    parser.add_argument("--batch_size_rf", type=int, default=1, help="Batch size if data='rf'.")
    parser.add_argument("--batch_size_f",  type=int, default=5, help="Batch size if data='f'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--root_raw", type=str, default="data/EUPPBench/raw", help="Path to raw data.")
    parser.add_argument("--root_processed", type=str, default="data/EUPPBench/processed", help="Path to processed data.")
    return parser.parse_args()

############################################################################
# 2. Reproducibility
############################################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

############################################################################
# 3. Evaluate Routines
############################################################################

@torch.no_grad()
def predict_model(model, loader, device):
    """
    Run inference on a dataset in plain PyTorch. Returns a single
    Tensor of predictions (concatenated).
    """
    model.eval()
    preds_list = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds_list.append(out.cpu())
    return torch.cat(preds_list, dim=0)

############################################################################
# 4. Main
############################################################################

def main():
    args = parse_args()

    # ----------------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------------
    os.makedirs(args.dir, exist_ok=True)
    log_dir = os.path.join(args.dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"eval_{args.data}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("========== Evaluation Script Started ==========")
    logger.info(f"Arguments: {args}")

    # ----------------------------------------------------------------------
    # Seed
    # ----------------------------------------------------------------------
    set_seed(args.seed)

    # ----------------------------------------------------------------------
    # Load Config
    # ----------------------------------------------------------------------
    config_path = os.path.join(args.dir, "params.json")
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config_dict = json.load(f)
    logger.info(f"Loaded config: {config_dict}")

    # ----------------------------------------------------------------------
    # Prepare dataset
    # ----------------------------------------------------------------------

    # We'll re-create a dataset to get the test set
    # e.g. if args.data == 'rf', we load 'test_rf', else 'test_f'
    split_name = "test_rf" if args.data == "rf" else "test_f"
    test_dataset = EUPPBench(
        root_raw=root_raw,
        root_processed=root_processed,
        leadtime=args.leadtime,
        max_dist=config_dict.get("max_dist", 100.0),
        split=split_name
    )
    logger.info(f"Test dataset => {len(test_dataset)} samples.")

    # Optionally apply split_graph if data=="f"
    if args.data == "f":
        # if you want sub-splitting logic
        new_list = []
        for d in test_dataset:
            subgs = split_graph(d, True)
            new_list.extend(subgs)
        test_dataset = new_list
        logger.info(f"After split_graph => #graphs: {len(test_dataset)}")

    # Build loader
    batch_size = args.batch_size_rf if args.data=="rf" else args.batch_size_f
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build a single target tensor
    # (One data.y per item in test_dataset)
    all_targets = []
    for d in test_dataset:
        all_targets.append(d.y)
    targets_tensor = torch.cat(all_targets, dim=0)

    # ----------------------------------------------------------------------
    # Load Checkpoints from dir
    # ----------------------------------------------------------------------
    ckpt_dir = os.path.join(args.dir, "models")
    if not os.path.isdir(ckpt_dir):
        logger.error(f"No 'models' subdir found at {ckpt_dir}")
        sys.exit(1)

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") or f.endswith(".pth")]
    if not ckpt_files:
        logger.error("No checkpoints found in %s", ckpt_dir)
        sys.exit(1)

    logger.info(f"Found {len(ckpt_files)} checkpoint(s) in '{ckpt_dir}'.")

    # ----------------------------------------------------------------------
    # Make model
    # ----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # infer input dims from the first item
    example_data = test_dataset[0]
    in_channels = example_data.x.shape[1]

    from torch.optim import AdamW

    # We'll ensemble predictions from multiple ckpts
    preds_ensemble = []

    for ckpt_name in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        logger.info(f"Loading checkpoint: {ckpt_path}")

        model = GNN(
        in_channels=in_channels,
        hidden_channels_gnn=config_dict["gnn_hidden"],
        out_channels_gnn=config_dict["gnn_hidden"],
        num_layers_gnn=config_dict["gnn_layers"],
        optimizer_class=AdamW,
        optimizer_params={"lr": config_dict["lr"]},
        loss=config_dict["loss"],
        grad_u=config_dict["grad_u"],
        u=config_dict["u"],
        xi=config_dict["xi"]
        ).to(device)

        # load state dict
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Predict
        preds = predict_model(model, test_loader, device)
        preds_ensemble.append(preds)

    # Stack & average
    if len(preds_ensemble) > 1:
        stacked = torch.stack(preds_ensemble, dim=0)  # shape [num_ckpts, #samples, out_dims]
        final_preds = stacked.mean(dim=0)             # shape [#samples, out_dims]
    else:
        final_preds = preds_ensemble[0]

    # Evaluate CRPS (or whatever your model uses)
    # The GNN has model.loss_fn
    # We'll reuse the last model instance for CRPS
    crps_value = model.loss_fn.crps(final_preds, targets_tensor)
    logger.info("========================================")
    logger.info(f"Final CRPS for data='{args.data}': {crps_value.item():.6f}")
    logger.info("========================================")

    # Optionally save CSV
    # If NormalCRPS => output_dims=2 => columns = ["tp6","mu","sigma"] (we can define logically)
    # We'll do a quick approach
    columns = ["tp6"] + [f"pred_{i}" for i in range(final_preds.shape[1])]
    combined_arr = torch.cat([targets_tensor.view(-1,1), final_preds], dim=1).cpu().numpy()
    df_out = pd.DataFrame(combined_arr, columns=columns)
    results_dir = os.path.join(args.dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{args.data}.csv")
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions to {csv_path}")

    # Summarize
    summary_path = os.path.join(results_dir, f"{args.data}_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"Data: {args.data}\n")
        f.write(f"Leadtime: {args.leadtime}\n")
        f.write(f"Final CRPS: {crps_value.item():.6f}\n")
    logger.info(f"Wrote summary to {summary_path}")
    logger.info("========== Evaluation Script Finished ==========")


if __name__ == "__main__":
    main()
