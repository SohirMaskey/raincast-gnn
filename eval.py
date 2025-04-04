# #!/usr/bin/env python3
# """
# eval.py

# Script to evaluate one or more trained GNN checkpoints
# using PyTorch Lightning. Logs to both console and file (no wandb).
# Includes seed setting for reproducibility.
# """

# import argparse
# import json
# import logging
# import os
# import sys
# import numpy as np
# import pandas as pd
# import torch

# import pytorch_lightning as L
# from pytorch_lightning import seed_everything
# from torch_geometric.loader import DataLoader
# from torch.optim import AdamW
# from dataclasses import dataclass

# from models.gnn import GNN
# from utils.data import (
#     split_graph
# )

# def parse_args():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(description="Evaluate an ensemble of graph-based model checkpoints.")
#     parser.add_argument(
#         "--data",
#         type=str,
#         default="rf",
#         choices=["rf", "f"],
#         help='Dataset to evaluate on. Either "rf" or "f".'
#     )
#     parser.add_argument(
#         "--leadtime",
#         type=str,
#         default="24h",
#         help='Lead time for evaluation, e.g. "24h", "72h", or "120h".'
#     )
#     parser.add_argument(
#         "--folder",
#         type=str,
#         required=True,
#         help="Folder containing 'params.json' and a 'models/' subfolder with .ckpt files."
#     )
#     parser.add_argument(
#         "--no_graph",
#         action="store_true",
#         help="Disable graph connectivity (should match training if used)."
#     )
#     parser.add_argument(
#         "--batch_size_rf",
#         type=int,
#         default=1,
#         help="Batch size to use when evaluating on 'rf' data."
#     )
#     parser.add_argument(
#         "--batch_size_f",
#         type=int,
#         default=5,
#         help="Batch size to use when evaluating on 'f' data."
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for reproducibility."
#     )

#     return parser.parse_args()


# def main():
#     """Main evaluation function with reproducibility in mind."""
#     args = parse_args()

#     # ------------------------------------------------------------------------------
#     # Set up logging
#     # ------------------------------------------------------------------------------
#     os.makedirs(args.folder, exist_ok=True)
#     log_file = os.path.join(args.folder, f"eval_{args.data}.log")

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[
#             logging.FileHandler(log_file, mode="w"),
#             logging.StreamHandler(sys.stdout)
#         ]
#     )
#     logger = logging.getLogger(__name__)

#     logger.info("========== Evaluation Script Started ==========")
#     logger.info("Command-line arguments: %s", args)

#     # ------------------------------------------------------------------------------
#     # Set random seed for reproducibility
#     # ------------------------------------------------------------------------------
#     seed_everything(args.seed, workers=True)
#     # torch.use_deterministic_algorithms(True)

#     # ------------------------------------------------------------------------------
#     # Load Config
#     # ------------------------------------------------------------------------------
#     json_path = os.path.join(args.folder, "params.json")
#     if not os.path.isfile(json_path):
#         logger.error("Config file not found: %s", json_path)
#         sys.exit(1)

#     with open(json_path, "r") as f:
#         config_dict = json.load(f)
#     logger.info("Loaded config from %s: %s", json_path, config_dict)

#     @dataclass
#     class DummyConfig:
#         pass

#     config = DummyConfig()
#     for key, value in config_dict.items():
#         setattr(config, key, value)

#     # ------------------------------------------------------------------------------
#     # Load Data
#     # ------------------------------------------------------------------------------
#     # dataframes = load_dataframes(leadtime=args.leadtime)

#     # dist_matrix = load_distances(dataframes["stations"])

#     # # Create Graphs
#     # graphs_train_rf, tests = normalize_features_and_create_graphs(
#     #     training_data=dataframes["train"],
#     #     valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
#     #     mat=dist_matrix,
#     #     max_dist=config.max_dist,
#     #     new_gnn=config.new_gnn
#     # )
#     # graphs_test_rf, graphs_test_f = tests

#     from torch.utils.data import random_split
#     from utils.dataset import EUPPBench

#     # 1. Create the train dataset (will download/unzip if needed, then process)
#     root_dir="/home/groups/ai/buelte/precip/Singapur-Trip-25/data"
#     root_processed="data/EUPPBench" 

#     graphs_train_rf = EUPPBench(root_raw=root_dir, root_processed=root_processed, leadtime="24h", split="train_rf")
#     graphs_test_rf = EUPPBench(root_raw=root_dir, root_processed=root_processed, leadtime="24h", split="test_rf")
#     graphs_test_f  = EUPPBench(root_raw=root_dir, root_processed=root_processed, leadtime="24h", split="test_f")

#     # Pick the relevant test set
#     if args.data == "rf":
#         graphs_test = graphs_test_rf
#     else:
#         graphs_test = graphs_test_f

#     graphs_test_labels = graphs_test.y

#     # If evaluating on "f" data and not using summary only, might need to split
#     if args.data == "f":
#         logger.info("Splitting graphs for 'f' data post-processing...")
#         splitted = [split_graph(g, config.new_gnn) for g in graphs_test]
#         graphs_test = [g for sublist in splitted for g in sublist]

#     # ------------------------------------------------------------------------------
#     # Data Loaders
#     # ------------------------------------------------------------------------------
#     logger.info("Creating DataLoaders...")
#     # Re-use the training set loader if needed for dummy passes
#     train_loader = DataLoader(
#         graphs_train_rf,
#         batch_size=config.batch_size,
#         shuffle=True,
#         generator=torch.Generator().manual_seed(args.seed)  # ensures consistent order
#     )

#     # If data == "rf", we default to batch_size_rf, else batch_size_f
#     test_batch_size = args.batch_size_rf if args.data == "rf" else args.batch_size_f
#     test_loader = DataLoader(graphs_test, batch_size=test_batch_size, shuffle=False)

#     # ------------------------------------------------------------------------------
#     # Determine Output Dimensions
#     # ------------------------------------------------------------------------------
#     if config.loss == "NormalCRPS":
#         output_dims = 2
#         columns = ["tp6", "mu", "sigma"]
#     elif config.loss == "MixedNormalCRPS":
#         output_dims = 3
#         columns = ["tp6", "mu", "sigma", "p"]
#     elif config.loss == "MixedLoss":
#         if str(config.grad_u).lower() in ["true", "1"]:
#             output_dims = 5
#             columns = ["tp6", "mu", "sigma", "p", "sigma_u", "u"]
#         else:
#             output_dims = 4
#             columns = ["tp6", "mu", "sigma", "p", "sigma_u"]
#     else:
#         # Default fallback
#         output_dims = 2
#         columns = ["tp6", "predA", "predB"]

#     # ------------------------------------------------------------------------------
#     # Model Ensemble
#     # ------------------------------------------------------------------------------
#     emb_dim = 20
#     # Example dimension; adapt to your dataset
#     in_channels = 55

#     ckpt_folder = os.path.join(args.folder, "models")
#     if not os.path.isdir(ckpt_folder):
#         logger.error("Expected 'models/' subfolder at %s", ckpt_folder)
#         sys.exit(1)

#     ckpt_files = [f for f in os.listdir(ckpt_folder) if f.endswith(".ckpt")]
#     if not ckpt_files:
#         logger.error("No .ckpt files found in %s", ckpt_folder)
#         sys.exit(1)

#     logger.info("Found %d checkpoint(s) in '%s'.", len(ckpt_files), ckpt_folder)

#     preds_list = []
#     for ckpt_name in ckpt_files:
#         ckpt_path = os.path.join(ckpt_folder, ckpt_name)
#         logger.info("Loading checkpoint: %s", ckpt_path)
#         checkpoint = torch.load(ckpt_path, map_location="cpu")

#         # Create model instance
#         model = GNN(
#             embedding_dim=emb_dim,
#             in_channels=in_channels,
#             hidden_channels_gnn=config.gnn_hidden,
#             out_channels_gnn=config.gnn_hidden,
#             num_layers_gnn=config.gnn_layers,
#             heads=config.heads,
#             hidden_channels_deepset=config.gnn_hidden,
#             optimizer_class=AdamW,
#             optimizer_params=dict(lr=config.lr),
#             loss=config.loss,
#             grad_u=config.grad_u,
#             u=config.u,
#             xi=config.xi,
#             no_graph=args.no_graph
#         )

#         # Dummy forward pass
#         dummy_batch = next(iter(train_loader))
#         model.forward(dummy_batch)

#         model.load_state_dict(checkpoint["state_dict"])

#         # Predict
#         trainer = L.Trainer(
#             log_every_n_steps=1,
#             accelerator="gpu",
#             devices=1,
#             enable_progress_bar=True,
#             logger=False,  # Disables Lightning default logging
#             # deterministic=True
#         )
#         predictions = trainer.predict(model=model, dataloaders=[test_loader])

#         # If data == "f" and not summary only, average across batch dimension
#         if args.data == "f":
#             # Each item in predictions is (batch_size, num_nodes, output_dims)
#             predictions = [
#                 x.reshape(test_batch_size, -1, output_dims).mean(axis=0)
#                 for x in predictions
#             ]

#         pred_tensor = torch.cat(predictions, dim=0)
#         preds_list.append(pred_tensor)

#     # Average across all checkpoints
#     stacked_preds = torch.stack(preds_list, dim=0)  # [num_ckpts, num_samples, output_dims]
#     final_preds = torch.mean(stacked_preds, dim=0)  # [num_samples, output_dims]

#     # ------------------------------------------------------------------------------
#     # Evaluate CRPS
#     # ------------------------------------------------------------------------------
#     targets_tensor = graphs_test_labels

#     # targets_tensor = torch.tensor(targets_df.tp6.values)
#     res = model.loss_fn.crps(final_preds, targets_tensor)

#     logger.info("========================================")
#     logger.info("Final CRPS for data='%s': %.6f", args.data, res.item())
#     logger.info("========================================")

#     # ------------------------------------------------------------------------------
#     # Save Predictions to CSV
#     # ------------------------------------------------------------------------------
#     combined = np.concatenate([targets_tensor.view(-1, 1), final_preds], axis=1)
#     results_df = pd.DataFrame(combined, columns=columns)
#     csv_path = os.path.join(args.folder, f"{args.data}_results.csv")
#     results_df.to_csv(csv_path, index=False)
#     logger.info("Saved predictions to %s", csv_path)

#     # ------------------------------------------------------------------------------
#     # Save Log Summary
#     # ------------------------------------------------------------------------------
#     summary_text = os.path.join(args.folder, f"{args.data}.txt")
#     with open(summary_text, "w") as f:
#         f.write(f"Data: {args.data}\n")
#         f.write(f"Leadtime: {args.leadtime}\n")
#         f.write(f"Final CRPS: {res.item():.6f}\n")
#     logger.info("Evaluation summary saved to %s", summary_text)
#     logger.info("========== Evaluation Script Finished ==========")


# if __name__ == "__main__":
#     main()


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

from dataclasses import dataclass
from torch_geometric.loader import DataLoader
from torch.optim import AdamW

# Example GNN
from models.gnn import GNN
# If we need custom logic
from utils.data import split_graph  # or remove if not needed

############################################################################
# 1. Argparse
############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a GNN model (plain PyTorch).")
    parser.add_argument("--data", type=str, default="rf", choices=["rf","f"], help="Test set to use: 'rf' or 'f'.")
    parser.add_argument("--leadtime", type=str, default="24h", help="Lead time, e.g. '24h'.")
    parser.add_argument("--folder", type=str, required=True, help="Folder w/ 'params.json' & 'models/' subfolder.")
    parser.add_argument("--no_graph", action="store_true", help="Disable graph connectivity.")
    parser.add_argument("--batch_size_rf", type=int, default=1, help="Batch size if data='rf'.")
    parser.add_argument("--batch_size_f",  type=int, default=5, help="Batch size if data='f'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
    os.makedirs(args.folder, exist_ok=True)
    log_dir = os.path.join(args.folder, "logs")
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
    config_path = os.path.join(args.folder, "params.json")
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config_dict = json.load(f)
    logger.info(f"Loaded config: {config_dict}")

    # ----------------------------------------------------------------------
    # Prepare dataset
    # ----------------------------------------------------------------------
    from utils.dataset import EUPPBench
    root_data = "/home/groups/ai/buelte/precip/Singapur-Trip-25/data"

    # We'll re-create a dataset to get the test set
    # e.g. if args.data == 'rf', we load 'test_rf', else 'test_f'
    split_name = "test_rf" if args.data == "rf" else "test_f"
    test_dataset = EUPPBench(
        root_raw=root_data,
        root_processed="data/EUPPBench",
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
    # Load Checkpoints from folder
    # ----------------------------------------------------------------------
    ckpt_folder = os.path.join(args.folder, "models")
    if not os.path.isdir(ckpt_folder):
        logger.error(f"No 'models' subfolder found at {ckpt_folder}")
        sys.exit(1)

    ckpt_files = [f for f in os.listdir(ckpt_folder) if f.endswith(".ckpt") or f.endswith(".pth")]
    if not ckpt_files:
        logger.error("No checkpoints found in %s", ckpt_folder)
        sys.exit(1)

    logger.info(f"Found {len(ckpt_files)} checkpoint(s) in '{ckpt_folder}'.")

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
        ckpt_path = os.path.join(ckpt_folder, ckpt_name)
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
    results_dir = os.path.join(args.folder, "results")
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
