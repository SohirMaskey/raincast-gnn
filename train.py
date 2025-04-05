#!/usr/bin/env python3
"""
train.py

Script to train a GNN model in plain PyTorch (no PyTorch Lightning).
Logs are written to a file and printed to console. Also includes
seed setting for reproducibility and simple checkpoint saving.
"""

import argparse
import json
import logging
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import random_split

from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from models.gnn import GNN
from utils.dataset import EUPPBench

############################################################################
# 1. Argparse
############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train a graph-based model (plain PyTorch).")
    parser.add_argument("--leadtime", type=str, default="24h", help="Lead time for the dataset (e.g. '24h').")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing params.json and for logs/checkpoints.")
    parser.add_argument("--run_id", type=str, required=True, help="Unique ID for this run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--root_raw", type=str, default="data/EUPPBench/raw", help="Path to raw data.")
    parser.add_argument("--root_processed", type=str, default="data/EUPPBench/processed", help="Path to processed data.")
    return parser.parse_args()

############################################################################
# 2. Reproducibility
############################################################################

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

############################################################################
# 3. Training & Validation Routines
############################################################################

def train_one_epoch(model, loader, optimizer, device, logger):
    """
    One epoch of training.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        # Forward
        preds = model(batch)
        loss = model.loss_fn.crps(preds, batch.y)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    logger.info(f"  [Train] Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate(model, loader, device, logger):
    """
    Validation loop. Returns average loss on the validation set.
    You can also compute CRPS or other metrics here.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            loss = model.loss_fn.crps(preds, batch.y)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    logger.info(f"  [Val] Loss: {avg_loss:.6f}")
    return avg_loss

############################################################################
# 4. Main
############################################################################

def main():
    args = parse_args()

    # ----------------------------------------------------------------------
    # Logging Setup
    # ----------------------------------------------------------------------
    os.makedirs(args.dir, exist_ok=True)
    log_dir = os.path.join(args.dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{args.run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("========== Training Script Started ==========")
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
        logger.error(f"Could not find params.json at: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info(f"Loaded config: {config}")

    # ----------------------------------------------------------------------
    # Create Dataset & Split
    # ----------------------------------------------------------------------
    # Example: using EUPPBench or your custom dataset
    
    train_dataset = EUPPBench(
        root_raw=args.root_raw,
        root_processed=args.root_processed, 
        leadtime=args.leadtime,
        max_dist=config.get("max_dist", 100.0),
        split="train_rf"  # e.g. train reforecasts
    )

    n_total = len(train_dataset)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    train_set, val_set = random_split(train_dataset, [n_train, n_val])
    logger.info(f"Dataset sizes => Train: {len(train_set)}, Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=config["batch_size"], shuffle=False)

    # ----------------------------------------------------------------------
    # Create Model
    # ----------------------------------------------------------------------
    logger.info("Creating model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    example_data = train_set[0]
    # Just an example of how in_channels might be computed
    in_channels = example_data.x.shape[1] #+ emb_dim - 1

    model = GNN(
        in_channels=in_channels,
        hidden_channels_gnn=config["gnn_hidden"],
        out_channels_gnn=config["gnn_hidden"],
        num_layers_gnn=config["gnn_layers"],
        optimizer_class=AdamW,
        optimizer_params={"lr": config["lr"]},
        loss=config["loss"],
        grad_u=config["grad_u"],
        u=config["u"],
        xi=config["xi"]
    ).to(device)

    # Check
    with torch.no_grad():
        sample_pred = model(example_data.to(device))

    optimizer = model.optimizer_class(model.parameters(), **model.optimizer_params)
    max_epochs = config["max_epochs"]

    # For checkpointing
    ckpt_dir = os.path.join(args.dir, "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = None

    # ----------------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------------
    logger.info(f"Starting training for {max_epochs} epochs...")
    for epoch in range(1, max_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{max_epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, logger)
        val_loss   = evaluate(model, val_loader, device, logger)

        # Checkpoint if improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(ckpt_dir, f"run_{args.run_id}-best.ckpt")
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"[Checkpoint] New best val_loss: {val_loss:.6f}. Saved to {best_ckpt_path}")

    logger.info("Training completed.")
    if best_ckpt_path:
        logger.info(f"Best checkpoint stored at {best_ckpt_path}")
    else:
        logger.info("No improvement found, no checkpoint saved.")

    logger.info("========== Training Script Finished ==========")


if __name__ == "__main__":
    main()
