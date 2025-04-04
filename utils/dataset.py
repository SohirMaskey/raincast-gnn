import os
import sys
import urllib.request
import zipfile
import shutil

import torch
from torch_geometric.data import InMemoryDataset, Data
from typing import List, Optional

# Use your existing code from data.py
# (We'll import the relevant functions here, or copy them inline.)
# For clarity, let's assume they're in a separate file named data_utils.py
from .data import (
    load_zarr_data_and_create_graphs
)

class EUPPBench(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for EUPPBench station-based forecast data.

    This dataset:
      1) Downloads and unzips raw data if not found locally,
      2) Loads data (pickled DataFrames or from Zarr),
      3) Processes them into PyG 'Data' objects for train_rf, test_rf, test_f,
      4) Saves each split to disk as a .pt file in the 'processed' folder.

    Instantiate it with a chosen 'split' to get the corresponding subset in memory.
    """
    url = "https://zenodo.org/records/7708362/files/EUPPBench-stations.zip?download=1"

    def __init__(
        self,
        root_raw: str,
        root_processed: str, 
        leadtime: str = "24h",
        max_dist: float = 100.0,
        split: str = "train_rf",
        transform=None,
        pre_transform=None
    ):
        """
        Args:
            root (str): Root directory where raw/ and processed/ folders will be created.
            leadtime (str): e.g. "24h", "72h", "120h".
            split (str): One of ["train_rf", "test_rf", "test_f"]. 
            transform: PyG transform.
            pre_transform: PyG pre_transform.
        """
        self.leadtime = leadtime
        self.max_dist = max_dist
        self.root_raw = root_raw
        self.root_processed = root_processed
        self.available_splits = ["train_rf", "test_rf", "test_f"]
        if split not in self.available_splits:
            raise ValueError(f"split must be one of {self.available_splits}, got {split}")
        self.split = split
        super().__init__(root_raw, transform, pre_transform)

        # The processed data is loaded from disk here:
        idx = self.available_splits.index(self.split)
        # We expect each split to have its own .pt file:
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_dir(self) -> str:
        return self.root_raw #os.path.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return self.root_processed #"data/EUPPBench" #os.path.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the names we expect in 'raw/'. 
        We'll store the downloaded zip there. 
        """
        return ["EUPPBench-stations.zip", "EUPPBench-stations"]

    @property
    def processed_file_names(self) -> List[str]:
        """
        We produce 3 .pt files, one for each split: train_rf, test_rf, test_f.
        The user can choose which one to load at init.
        """
        # We'll name them with the leadtime prefix
        lt = self.leadtime
        return [
            f"EUPPBench_{lt}_train_rf.pt",
            f"EUPPBench_{lt}_test_rf.pt",
            f"EUPPBench_{lt}_test_f.pt"
        ]

    def download(self):
        """
        Downloads the EUPPBench-stations.zip from Zenodo if not already in raw_dir,
        then unzips it into the same location.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        zip_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_path = os.path.join(self.raw_dir, self.raw_file_names[1])

        if os.path.exists(data_path):
            print("[INFO] Data directory already exists; skipping download.")
            return
        # If the zip isn't present, download it
        if not os.path.exists(zip_path):
            print(f"[INFO] Downloading EUPPBench data from {self.url} ...")
            try:
                urllib.request.urlretrieve(self.url, zip_path)
            except Exception as e:
                print(f"[ERROR] Failed to download data: {e}")
                sys.exit(1)

        # Unzip if not already unzipped
        extracted_dir = os.path.join(self.raw_dir, "EUPPBench-stations")
        if not os.path.exists(extracted_dir):
            print(f"[INFO] Unzipping data to {extracted_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
        else:
            print("[INFO] Data directory already exists; skipping unzip.")

    def process(self):
        """
        1. Load data (via load_dataframes, load_distances, normalize_features_and_create_graphs),
        2. Create PyG 'Data' objects,
        3. Collate them and save each split to disk as .pt files in self.processed_dir.
        """
        # Step 1) Load dataframes
        # We'll call 'load_dataframes' from your existing logic:
        print("[INFO] Processing EUPPBench dataset...")
        # dataframes = load_dataframes(leadtime=self.leadtime)  # => dict with train, test_rf, test_f, stations
        
        dataframes = load_zarr_data_and_create_graphs(
            zarr_path=os.path.join(self.raw_dir, "EUPPBench-stations"), 
            leadtime=self.leadtime, 
            max_dist=self.max_dist
        )
        dist_matrix = dataframes["stations"]

        # # Step 2) Create graph data
        # graphs_train_rf, tests = normalize_features_and_create_graphs(
        #     training_data=dataframes["train"],
        #     valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
        #     mat=dist_matrix,
        #     max_dist=100.0,  # or from config if you prefer
        #     new_gnn=False
        # )
        # graphs_test_rf, graphs_test_f = tests  # 'tests' is a list of length 2
        graphs_train_rf = dataframes["train"]
        graphs_test_rf = dataframes["test_rf"]
        graphs_test_f  = dataframes["test_f"]

        # Step 3) We now have 3 lists of PyG Data objects:
        #   graphs_train_rf
        #   graphs_test_rf
        #   graphs_test_f

        # If you want transforms, apply them now:
        if self.pre_transform is not None:
            graphs_train_rf = [self.pre_transform(g) for g in graphs_train_rf]
            graphs_test_rf = [self.pre_transform(g) for g in graphs_test_rf]
            graphs_test_f  = [self.pre_transform(g) for g in graphs_test_f]

        # Convert each list of Data objects to a big InMemoryDataset
        data_splits = {
            "train_rf": graphs_train_rf,
            "test_rf": graphs_test_rf,
            "test_f": graphs_test_f,
        }

        os.makedirs(self.processed_dir, exist_ok=True)

        # Save each split to its .pt file
        for i, split_name in enumerate(["train_rf", "test_rf", "test_f"]):
            data_list = data_splits[split_name]
            if len(data_list) == 0:
                # Edge case: if a split is empty, skip it or store an empty object
                print(f"[WARNING] {split_name} split is empty.")
            data, slices = self.collate(data_list)
            filename = self.processed_file_names[i]  # e.g., "EUPPBench_24h_train_rf.pt"
            path = os.path.join(self.processed_dir, filename)
            torch.save((data, slices), path)

        print("[INFO] Finished processing and saving splits.")

    def __len__(self):
        # Overridden by InMemoryDataset. We'll rely on self.slices to determine length.
        return super().__len__()
