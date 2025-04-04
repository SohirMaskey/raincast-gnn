# import geopy.distance
# import numpy as np
# import os
# import pandas as pd
# import torch
# import torch_geometric
# import xarray

# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler
# from torch_geometric.data import Data
# from typing import DefaultDict, Tuple, List, Union


# class ZarrLoader:
#     """
#     A class for loading data from Zarr files.

#     Args:
#         data_path (str): The path to the data directory.

#     Attributes:
#         data_path (str): The path to the data directory.
#         leadtime (pd.Timedelta): The lead time for the forecasts.
#         countries (List[str]): The list of countries to load data for.
#         features (List[str]): The list of features to load.

#     Methods:
#         get_stations(arr: xarray.Dataset) -> pd.DataFrame:
#             Get the stations information from the dataset.

#         load_data(leadtime: str = "24h", countries: Union[str, List[str]] = "all",
#         features: Union[str, List[str]] = "all")
#         -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset, xarray.Dataset]:
#             Load the data from Zarr files.

#         validate_stations() -> bool:
#             Validate if the station IDs match between forecasts and reforecasts.
#     """

#     def __init__(self, data_path: str) -> None:
#         self.data_path = data_path

#     def get_stations(self, arr: xarray.Dataset) -> pd.DataFrame:
#         """
#         Get the stations information from the dataset.

#         Args:
#             arr (xarray.Dataset): The dataset containing station information.

#         Returns:
#             pd.DataFrame: The dataframe containing station information.
#         """
#         stations = pd.DataFrame(
#             {
#                 "station_id": arr.station_id.values,
#                 "lat": arr.station_latitude.values,
#                 "lon": arr.station_longitude.values,
#                 "altitude": arr.station_altitude.values,
#                 "name": arr.station_name.values,
#             }
#         )
#         stations = stations.sort_values("station_id").reset_index(drop=True)
#         return stations

#     def load_data(
#         self, leadtime: str = "24h", countries: Union[str, List[str]] = "all", features: Union[str, List[str]] = "all"
#     ) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset, xarray.Dataset]:
#         """
#         Load data for the specified lead time, countries, and features.

#         Args:
#             leadtime (str): The lead time for the forecasts and reforecasts. Default is "24h".
#             countries (Union[str, List[str]]): The countries for which to load the data. Default is "all".
#             features (Union[str, List[str]]): The features to load. Default is "all".

#         Returns:
#             Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset, xarray.Dataset]:
#             A tuple containing the following datasets:
#                 - df_f: The forecasts dataset.
#                 - df_f_target: The targets for the forecasts dataset.
#                 - df_rf: The reforecasts dataset.
#                 - df_rf_target: The targets for the reforecasts dataset.
#         """
#         self.leadtime = pd.Timedelta(leadtime)

#         if countries == "all":
#             print("[INFO] Loading data for all countries")
#             self.countries = ["austria", "belgium", "france", "germany", "netherlands"]
#         elif isinstance(countries, list):
#             print(f"[INFO] Loading data for {countries}")
#             self.countries = countries
#         else:
#             raise ValueError("countries must be a list of strings or 'all'")

#         if features == "all":
#             print("[INFO] Loading all features")
#             self.features = ["number"] + [
#                 "station_id",
#                 "time",
#                 "cape",
#                 "model_orography",
#                 "sd",
#                 "station_altitude",
#                 "station_latitude",
#                 "station_longitude",
#                 "stl1",
#                 "swvl1",
#                 "t2m",
#                 "tcc",
#                 "tcw",
#                 "tcwv",
#                 "u10",
#                 "u100",
#                 "v10",
#                 "v100",
#                 "vis",
#                 "cp6",
#                 "mn2t6",
#                 "mx2t6",
#                 "p10fg6",
#                 "slhf6",
#                 "sshf6",
#                 "ssr6",
#                 "ssrd6",
#                 "str6",
#                 "strd6",
#                 "tp6",
#                 "z",
#                 "q",
#                 "u",
#                 "v",
#                 "t",
#             ]
#         elif isinstance(features, list):
#             print(f"[INFO] Loading features: {features}")
#             self.features = ["number"] + features
#         else:
#             raise ValueError("features must be a list of strings or 'all'")

#         # Load Data from Zarr ####
#         forecasts_all_countries = []
#         reforecasts_all_countries = []

#         targets_f_all_countries = []
#         targets_rf_all_countries = []
#         for country in self.countries:
#             print(f"[INFO] Loading data for {country}")
#             # Forecasts
#             f_surface_xr = xarray.open_zarr(f"{self.data_path}/stations_ensemble_forecasts_surface_{country}.zarr")
#             f_surface_pp_xr = xarray.open_zarr(
#                 f"{self.data_path}/stations_ensemble_forecasts_surface_postprocessed_{country}.zarr"
#             )
#             f_pressure_500_xr = xarray.open_zarr(
#                 f"{self.data_path}/stations_ensemble_forecasts_pressure_500_{country}.zarr"
#             )
#             f_pressure_700_xr = xarray.open_zarr(
#                 f"{self.data_path}/stations_ensemble_forecasts_pressure_700_{country}.zarr"
#             )
#             f_pressure_850_xr = xarray.open_zarr(
#                 f"{self.data_path}/stations_ensemble_forecasts_pressure_850_{country}.zarr"
#             )
#             f_obs_xr = xarray.open_zarr(f"{self.data_path}/stations_forecasts_observations_surface_postprocessed_{country}.zarr")
#             forecasts = [f_surface_xr, f_surface_pp_xr, f_pressure_500_xr, f_pressure_700_xr, f_pressure_850_xr]

#             # Reforecasts
#             rf_surface_xr = xarray.open_zarr(f"{self.data_path}/stations_ensemble_reforecasts_surface_{country}.zarr")
#             rf_surface_pp_xr = xarray.open_zarr(
#                 f"{self.data_path}/stations_ensemble_reforecasts_surface_postprocessed_{country}.zarr"
#             )
#             rf_pressure_500_xr = xarray.open_zarr(
#                 f"{self.data_path}/stations_ensemble_reforecasts_pressure_500_{country}.zarr"
#             )
#             rf_pressure_700_xr = xarray.open_zarr(
#                 f"{self.data_path}/stations_ensemble_reforecasts_pressure_700_{country}.zarr"
#             )
#             rf_pressure_850_xr = xarray.open_zarr(
#                 f"{self.data_path}/stations_ensemble_reforecasts_pressure_850_{country}.zarr"
#             )
#             rf_obs_xr = xarray.open_zarr(f"{self.data_path}/stations_reforecasts_observations_surface_postprocessed_{country}.zarr")
#             reforecasts = [rf_surface_xr, rf_surface_pp_xr, rf_pressure_500_xr, rf_pressure_700_xr, rf_pressure_850_xr]

#             forecasts = [forecast.drop_vars("valid_time").squeeze(drop=True) for forecast in forecasts]
#             reforecasts = [reforecast.drop_vars("valid_time").squeeze(drop=True) for reforecast in reforecasts]

#             forecasts = xarray.merge(forecasts).sel(step=self.leadtime)
#             reforecasts = xarray.merge(reforecasts).sel(step=self.leadtime)

#             forecasts_all_countries.append(forecasts)
#             reforecasts_all_countries.append(reforecasts)

#             targets_f = f_obs_xr.squeeze(drop=True).sel(step=self.leadtime)
#             targets_rf = rf_obs_xr.squeeze(drop=True).sel(step=self.leadtime)

#             targets_f_all_countries.append(targets_f)
#             targets_rf_all_countries.append(targets_rf)

#         forecasts = xarray.concat(forecasts_all_countries, dim="station_id")
#         reforecasts = xarray.concat(reforecasts_all_countries, dim="station_id")

#         targets_f = xarray.concat(targets_f_all_countries, dim="station_id")
#         targets_rf = xarray.concat(targets_rf_all_countries, dim="station_id")

#         forecasts = forecasts.drop_vars(
#             ["model_altitude", "model_land_usage", "model_latitude", "model_longitude", "station_land_usage", "step"]
#         )
#         reforecasts = reforecasts.drop_vars(
#             ["model_altitude", "model_land_usage", "model_latitude", "model_longitude", "station_land_usage", "step"]
#         )
#         print(
#             f"[INFO] Data loaded successfully. Forecasts shape:\
#             {forecasts.tp6.shape}, Reforecasts shape: {reforecasts.tp6.shape}"
#         )
#         # Extract Stations ####
#         self.stations_f = self.get_stations(forecasts)
#         self.stations_rf = self.get_stations(reforecasts)

#         # Turn into pandas Dataframe ####
#         df_f = (
#             forecasts.to_dataframe()
#             .reorder_levels(["time", "number", "station_id"])
#             .sort_index(level=["time", "number", "station_id"])
#             .reset_index()
#         )
#         df_f_target = (
#             targets_f.tp6.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name", "step"])
#             .to_dataframe()
#             .reorder_levels(["time", "station_id"])
#             .sort_index(level=["time", "station_id"])
#             .reset_index()
#         )

#         df_rf = reforecasts.to_dataframe().reset_index()
#         df_rf_target = (
#             targets_rf.tp6.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name", "step"])
#             .to_dataframe()
#             .reset_index()
#         )

#         df_rf["time"] = df_rf["time"] - df_rf["year"].apply(lambda x: pd.Timedelta((21 - x) * 365, unit="day"))
#         df_rf_target["time"] = df_rf_target["time"] - df_rf_target["year"].apply(
#             lambda x: pd.Timedelta((21 - x) * 365, unit="day")  # ! 21 or 20 years of reforecasts
#         )

#         df_rf = df_rf.drop(columns=["year"]).reindex(columns=df_f.columns).sort_values(["time", "number", "station_id"])
#         df_rf_target = (
#             df_rf_target.drop(columns=["year"]).reindex(columns=df_f_target.columns).sort_values(["time", "station_id"])
#         )

#         # Turn Station IDs into a Station Index starting from 0
#         station_ids = df_f.station_id.unique()
#         id_to_index = {station_id: i for i, station_id in enumerate(station_ids)}

#         df_f["station_id"] = df_f["station_id"].apply(lambda x: id_to_index[x])
#         df_f_target["station_id"] = df_f_target["station_id"].apply(lambda x: id_to_index[x])
#         df_rf["station_id"] = df_rf["station_id"].apply(lambda x: id_to_index[x])
#         df_rf_target["station_id"] = df_rf_target["station_id"].apply(lambda x: id_to_index[x])

#         # Transform precipitation
#         # Transform units to mm and apply log(x+0.01) transformation
#         df_f_target["tp6"] = np.log(df_f_target["tp6"].clip(lower = 0)*1000 + 0.01)
#         df_rf_target["tp6"] = np.log(df_rf_target["tp6"].clip(lower = 0)*1000 + 0.01)

#         df_f["tp6"] = np.log(df_f["tp6"].clip(lower = 0)*1000 + 0.01)
#         df_rf["tp6"] = np.log(df_rf["tp6"].clip(lower = 0)*1000 + 0.01)

#         # Cut features ####
#         df_f = df_f[self.features]
#         df_rf = df_rf[self.features]

#         return df_f, df_f_target, df_rf, df_rf_target

#     def validate_stations(self):
#         return (self.stations_f.station_id == self.stations_rf.station_id).all()


# def load_dataframes(
#     leadtime: str,
# ) -> DefaultDict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
#     """Load the dataframes for training, testing on reforecasts, and testing on forecasts either from Zarr
#     or as a pandas Dataframe. If the dataframes do not exist as pandas Dataframe,
#     they are created and saved to disk so future loading is faster.

#     Args:
#         mode (str): The mode of the script, can be "train", "eval", or "hyperopt".
#         leadtime (str): The leadtime of the predictions, can be "24h", "72h", or "120h".

#     Returns:
#         DefaultDict[str, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary containing the dataframes for
#         training, validation and testing.

#     """

#     # Load Data ######################################################################
#     DATA_FOLDER = f"/home/groups/ai/buelte/precip/Singapur-Trip-25/data/dataframes_{leadtime}"
#     #DATA_FOLDER = f"data/dataframes_{leadtime}"
#     res = defaultdict(lambda: None)

#     DATA_FOLDER = os.path.join(DATA_FOLDER, "final_train")
#     # Training data
#     TRAIN_RF_PATH = os.path.join(DATA_FOLDER, "train_rf_final.pkl")
#     TRAIN_RF_TARGET_PATH = os.path.join(DATA_FOLDER, "train_rf_target_final.pkl")
#     # Test on Reforceasts
#     TEST_RF_PATH = os.path.join(DATA_FOLDER, "valid_rf_final.pkl")
#     TEST_RF_TARGET_PATH = os.path.join(DATA_FOLDER, "valid_rf_target_final.pkl")
#     # Test on Forecasts
#     TEST_F_PATH = os.path.join(DATA_FOLDER, "test_f_final.pkl")
#     TEST_F_TARGET_PATH = os.path.join(DATA_FOLDER, "test_f_target_final.pkl")

#     STATIONS_PATH = os.path.join(DATA_FOLDER, "stations.pkl")

#     # Check if the files exist
#     if (
#         os.path.exists(TRAIN_RF_PATH)
#         and os.path.exists(TRAIN_RF_TARGET_PATH)
#         and os.path.exists(TEST_RF_PATH)
#         and os.path.exists(TEST_RF_TARGET_PATH)
#         and os.path.exists(TEST_F_PATH)
#         and os.path.exists(TEST_F_TARGET_PATH)
#         and os.path.exists(STATIONS_PATH)
#     ):

#         print("[INFO] Dataframes exist. Will load pandas dataframes.")
#         train_rf = pd.read_pickle(TRAIN_RF_PATH)
#         train_rf_target = pd.read_pickle(TRAIN_RF_TARGET_PATH)
#         #print("Mean: ", np.nonzero(np.exp(train_rf_target.loc[:, 'tp6']) -0.01 ).mean())
#         #print("Variance: ",  np.nonzero(np.exp(train_rf.loc[:, 'tp6']) -0.01 ).std())
#         #print("Variance: ",  np.nonzero(np.exp(train_rf.loc[:, 'tp6']) -0.01).median())

#         test_rf = pd.read_pickle(TEST_RF_PATH)
#         test_rf_target = pd.read_pickle(TEST_RF_TARGET_PATH)

#         test_f = pd.read_pickle(TEST_F_PATH)
#         test_f_target = pd.read_pickle(TEST_F_TARGET_PATH)
        
#         # print("Variance RF: ", test_rf.loc[:, 'tp6'].std())
#         # print("Variance F: ", test_f.loc[:, 'tp6'].std())
#         # exit()

#         stations_f = pd.read_pickle(STATIONS_PATH)

#     else:
#         print("[INFO] Data files not found, will load from zarr.")
#         loader = ZarrLoader("/home/groups/ai/buelte/precip/Singapur-Trip-25/data/EUPPBench-stations")
#         print("[INFO] Loading data...")
#         df_f, df_f_target, df_rf, df_rf_target = loader.load_data(
#             leadtime=leadtime, countries="all", features="all"
#         )
#         assert loader.validate_stations(), "Stations in forecasts and reforecasts do not match."
#         stations_f = loader.stations_f

#         # Split the data
#         # Test 2014-2017 # 4 years (Forecasts)
#         # Test2 2014-15 # 2 years (Reforecasts)
#         # Now train with full data
#         # Train 1997-2013 # 13 years (Reforecasts)
#         train_cutoff = pd.Timestamp("2014-01-01")
#         train_rf = df_rf.loc[df_rf["time"] < train_cutoff, :]
#         train_rf_target = df_rf_target.loc[df_rf_target["time"] < train_cutoff, :]

#         test_rf = df_rf.loc[(df_rf["time"] >= train_cutoff), :]
#         test_rf_target = df_rf_target.loc[(df_rf_target["time"] >= train_cutoff), :]

#         test_f = df_f
#         test_f_target = df_f_target

#         if not os.path.exists(DATA_FOLDER):
#             os.makedirs(DATA_FOLDER)
#         print("[INFO] Saving dataframes to disk...")
#         train_rf.to_pickle(TRAIN_RF_PATH)
#         train_rf_target.to_pickle(TRAIN_RF_TARGET_PATH)

#         test_rf.to_pickle(TEST_RF_PATH)
#         test_rf_target.to_pickle(TEST_RF_TARGET_PATH)

#         test_f.to_pickle(TEST_F_PATH)
#         test_f_target.to_pickle(TEST_F_TARGET_PATH)

#         stations_f.to_pickle(STATIONS_PATH)

#     res["train"] = (train_rf, train_rf_target)
#     res["test_rf"] = (test_rf, test_rf_target)
#     res["test_f"] = (test_f, test_f_target)
#     res["stations"] = stations_f
#     return res

# def load_stations(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
#     """Create a DataFrame containing station-specific data from the input DataFrame.

#     :param df: The DataFrame created by load_data.
#     :type df: pd.DataFrame

#     Returns:
#         Tuple[pd.DataFrame, np.ndarray]]: Dataframe with stations and numpy array with station positions
#     """
#     stations = df.groupby(by="station")[["lat", "lon", "alt", "orog"]].first().reset_index()
#     stations.station = pd.to_numeric(stations.station, downcast="integer")

#     postions_matrix = np.array(stations[["station", "lon", "lat"]])
#     return stations, postions_matrix


# def load_distances(stations: pd.DataFrame) -> np.ndarray:
#     """Load the distance matrix from file if it exists, otherwise compute it and save it to file.

#     Args:
#         stations (pd.DataFrame): The stations dataframe.

#     Returns:
#         np.ndarray: The distance matrix.
#     """
#     # Load Distances #################################################################
#     if os.path.exists("data/distances_EUPP.npy"):
#         print("[INFO] Loading distances from file...")
#         mat = np.load("data/distances_EUPP.npy")
#     elif os.path.exists("/home/groups/ai/buelte/precip/Singapur-Trip-25/data/distances_EUPP.npy"):
#         print("[INFO] Loading distances from file...")
#         mat = np.load("/home/groups/ai/buelte/precip/Singapur-Trip-25/data/distances_EUPP.npy")
#     else:
#         print("[INFO] Computing distances...")
#         mat = compute_dist_matrix(stations)
#         np.save("/home/groups/ai/buelte/precip/Singapur-Trip-25/data/distances_EUPP.npy", mat)
#     return mat


# def dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
#     """
#     Returns the distance between two stations in kilometers using the WGS-84 ellipsoid.

#     :param lat1: Latitude of the first station.
#     :type lat1: float
#     :param lat2: Latitude of the second station.
#     :type lat2: float
#     :param lon1: Longitude of the first station.
#     :type lon1: float
#     :param lon2: Longitude of the second station.
#     :type lon2: float

#     :return: The distance between the two stations in kilometers.
#     :rtype: float
#     """
#     return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km


# def compute_dist_matrix(df: pd.DataFrame) -> np.array:
#     """
#     Returns a distance matrix between stations.

#     :param df: dataframe with stations

#     :return: distance matrix
#     :rtype: np.array
#     """
#     coords_df = df[["lat", "lon"]].copy()

#     # create numpy arrays for latitudes and longitudes
#     latitudes = np.array(coords_df["lat"])
#     longitudes = np.array(coords_df["lon"])

#     # create a meshgrid of latitudes and longitudes
#     lat_mesh, lon_mesh = np.meshgrid(latitudes, longitudes)

#     # calculate distance matrix using vectorized distance function
#     distance_matrix = np.vectorize(dist_km)(lat_mesh, lon_mesh, lat_mesh.T, lon_mesh.T)
#     return distance_matrix

# def build_edge_index_and_attr(dist_mat: np.ndarray, max_dist: float) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Build edge_index and edge_attr from a distance matrix by:
#     1) Setting diagonal to inf (removing self loops).
#     2) Selecting edges within 'max_dist'.
#     3) Normalizing distances and optionally inverting them.
#     4) Adding self loops with a fixed attribute = 1.
#     """
#     D = dist_mat.copy()
#     np.fill_diagonal(D, np.inf)  # remove self loops
#     # Select edges
#     mask = (D <= max_dist)
#     row, col = np.where(mask)
#     valid_distances = D[row, col]
#     max_val = valid_distances.max() if valid_distances.max() > 0 else 1.0

#     # Normalize and invert => smaller distances => larger edge weight
#     norm_d = np.power(valid_distances / max_val, -1)

#     # Build PyG tensors
#     edge_index = torch.tensor([row, col], dtype=torch.long)
#     edge_attr = torch.tensor(norm_d, dtype=torch.float32).unsqueeze(1)

#     # Add self loops
#     N = dist_mat.shape[0]
#     self_loops = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2,1)  # shape [2, N]
#     self_attr = torch.ones((N, 1), dtype=torch.float32)

#     edge_index = torch.cat([edge_index, self_loops], dim=1)
#     edge_attr = torch.cat([edge_attr, self_attr], dim=0)
#     return edge_index, edge_attr

# def create_graphs_per_time(df_features, df_targets, dist_mat, max_dist) -> List[Data]:
#     """
#     Creates one PyG Data object per unique time stamp. For each time step:

#     - We assume each station appears for each ensemble member (field 'number').
#     - We group all ensemble members' features into a [N, E, F_pred] tensor, stored in data.ensemble.
#     - We pick 'extra' features from the first row per station as data.x, shape [N, F_extra].
#     - The target is shape [N], one per station, stored as data.y.

#     The function returns a list of Data objects.

#     Args:
#         df_features: DataFrame with columns [station_id, time, number, ... features ...]
#         df_targets: DataFrame with columns [time, station_id, tp6, ...], same times/stations
#         dist_mat: np.ndarray [N, N], distance matrix for stations
#         max_dist: float, distance threshold in km
#     """
#     edge_index, edge_attr = build_edge_index_and_attr(dist_mat, max_dist)

#     graphs = []
#     times = np.sort(df_features["time"].unique())
#     for t in times:
#         day_df = df_features[df_features["time"] == t]
#         day_target_df = df_targets[df_targets["time"] == t].sort_values("station_id")
        
#         stations = np.sort(day_df["station_id"].unique())
#         N = len(stations)
#         E = day_df["number"].nunique()

#         # The columns we treat as predictions (ens or others)
#         exclude_cols = ["station_id", "time", "number"]
#         pred_cols = [c for c in day_df.columns if c not in exclude_cols]

#         # Extra (non-ensemble) features => first occurrence per station
#         # shape => [N, F_extra]
#         extra_feats = day_df.groupby("station_id").first()[pred_cols].to_numpy()
#         extra_feats_t = torch.tensor(extra_feats, dtype=torch.float32)

#         # Ensemble predictions => shape [N, E, F_pred]
#         sorted_df = day_df.sort_values(["station_id", "number"])
#         ens_array = sorted_df[pred_cols].to_numpy().reshape(N, E, -1)
#         ens_tensor = torch.tensor(ens_array, dtype=torch.float32)

#         # Targets => 1 per station
#         y_np = day_target_df["tp6"].to_numpy()  # shape [N]
#         y_t = torch.tensor(y_np, dtype=torch.float32)

#         data = Data(
#             x=extra_feats_t,    # shape [N, F_extra]
#             ensemble=ens_tensor,  # shape [N, E, F_pred]
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             y=y_t
#         )
#         data.batch = torch.zeros(N, dtype=torch.long)  # all nodes => batch idx 0
#         data.timestamp = t
#         graphs.append(data)
#     return graphs


# def normalize_features_and_create_graphs(
#     training_data: pd.DataFrame,
#     valid_test_data: List[Tuple[pd.DataFrame]],
#     mat: np.ndarray,
#     max_dist: float,
#     new_gnn=False
# ) -> Tuple[List[torch_geometric.data.Data], List[List[torch_geometric.data.Data]]]:
#     """
#     Normalize the features in the training data and create graph data.

#     Args:
#         training_data (pd.DataFrame): The training data.
#         valid_test_data (List[Tuple[pd.DataFrame]]): The validation and test data. Each Tuple consists of the Features
#         and Targets.
#         mat (np.ndarray): The distance matrix.
#         max_dist (float): The maximum distance.

#     Returns:
#         Tuple[List[torch_geometric.data.Data], List[List[torch_geometric.data.Data]]]:
#         A tuple containing the graph data for the training data and the validation/test data.
#     """

#     if new_gnn:
#         # 1) Identify features to normalize
#         train_df, train_target = training_data
#         # Exclude columns that are not numeric or that you do not want to scale
#         exclude_cols = ["station_id", "time", "number"]
#         features_to_normalize = [c for c in train_df.columns if c not in exclude_cols]

#         # Fit scaler on train
#         scaler = StandardScaler()
#         train_df = train_df.copy()
#         train_df.loc[:, features_to_normalize] = scaler.fit_transform(train_df[features_to_normalize])

#         # Apply to valid/test sets
#         vt_dataframes = []
#         for (dfv, dfv_target) in valid_test_data:
#             dfv = dfv.copy()
#             dfv.loc[:, features_to_normalize] = scaler.transform(dfv[features_to_normalize])
#             vt_dataframes.append((dfv, dfv_target))

#         # 2) Add cyclical time features for train & val/test
#         def add_cyclical_features(df: pd.DataFrame):
#             # If dt is not already datetime, parse it.
#             if not np.issubdtype(df["time"].dtype, np.datetime64):
#                 df["time"] = pd.to_datetime(df["time"])
#             df["cos_doy"] = np.cos(2 * np.pi * df["time"].dt.dayofyear / 365)
#             df["sin_doy"] = np.sin(2 * np.pi * df["time"].dt.dayofyear / 365)

#         add_cyclical_features(train_df)
#         for (dfv, _) in vt_dataframes:
#             add_cyclical_features(dfv)

#         # 3) Build the train graph list
#         graphs_train = create_graphs_per_time(train_df, train_target, mat, max_dist)

#         # 4) Build the val/test graph lists
#         graphs_valid_test = []
#         for (dfv, dfv_target) in vt_dataframes:
#             graphs_v = create_graphs_per_time(dfv, dfv_target, mat, max_dist)
#             graphs_valid_test.append(graphs_v)

#         return graphs_train, graphs_valid_test
#     else:
#         # Normalize Features ############################################################
#         # Select the features to normalize
#         print("[INFO] Normalizing features...")
#         train_rf = training_data[0]
#         features_to_normalize = [col for col in train_rf.columns if col not in ["station_id", "time", "number"]]

#         # Create a MinMaxScaler object
#         scaler = StandardScaler()

#         # Fit and transform the selected features
#         train_rf.loc[:, features_to_normalize] = scaler.fit_transform(train_rf[features_to_normalize]).astype("float32")

#         train_rf.loc[:, ["cos_doy"]] = np.cos(2 * np.pi * train_rf["time"].dt.dayofyear / 365)
#         train_rf.loc[:, ["sin_doy"]] = np.sin(2 * np.pi * train_rf["time"].dt.dayofyear / 365)

#         for features, targets in valid_test_data:
#             features.loc[:, features_to_normalize] = scaler.transform(features[features_to_normalize]).astype("float32")
#             features.loc[:, ["cos_doy"]] = np.cos(2 * np.pi * features["time"].dt.dayofyear / 365)
#             features.loc[:, ["sin_doy"]] = np.sin(2 * np.pi * features["time"].dt.dayofyear / 365)

#         # Create Graph Data ##############################################################
#         # ! a conversion from kelvin to celsius is also done in create_multigraph
#         print("[INFO] Creating graph data...")
#         graphs_train_rf = create_multigraph(df=train_rf, df_target=training_data[1], distances=mat, max_dist=max_dist)

#         test_valid = []
#         for features, targets in valid_test_data:
#             graphs_valid_test = create_multigraph(df=features, df_target=targets, distances=mat, max_dist=max_dist)
#             test_valid.append(graphs_valid_test)

#         return graphs_train_rf, test_valid


# def split_graph(graph, new_gnn=False) -> List[torch_geometric.data.Data]:
#     """Splits a graph which is created using 51 ensemble members into 5 subgraphs,
#     each containing 10 or 11 ensemble members.

#     Args:
#         graph (torch_geometric.data.Data): the graph to be split
#     """
#     if new_gnn:
#         sets = [
#         0,
#         10,
#         20,
#         30,
#         40,
#         50,
#         ]  # Each set contains a list of station indices corresponding to 10 (or 11) ensemble members
#         graphs = []
#         # print(graph.ensemble.shape)
#         # exit()
#         for i in range(0,5):
#             graph_copy = graph.clone()
#             graph_copy.ensemble = graph_copy.ensemble[:,sets[i]:sets[i+1],:]
#             graphs.append(graph_copy)
#         return graphs
#     else:
#         perm = torch.randperm(51) * 122  # First node of each ensemble member
#         index = perm[:, None] + torch.arange(122)  # Add the node indices to each ensemble member

#         set1 = index[:10]
#         set2 = index[10:20]
#         set3 = index[20:30]
#         set4 = index[30:40]
#         set5 = index[40:]  # Has 11 elements

#         sets = [
#             set1,
#             set2,
#             set3,
#             set4,
#             set5,
#         ]  # Each set contains a list of station indices corresponding to 10 (or 11) ensemble members
#         graphs = []
#         for s in sets:
#             graphs.append(graph.subgraph(s.flatten()))
#         return graphs


# def shuffle_features(xs: torch.Tensor, feature_permute_idx: List[int]) -> torch.Tensor:
#     """Shuffle a tensor of the shape [T, N, F] first along the T dimension and then along the N dimension

#     Args:
#         xs (torch.Tensor): [T, N, F]
#         feature_permute_idx (List[int]): indices of the features to permute
#         (can be used to permute certain features together)

#     Returns:
#         torch.tensor: the shuffled tensor
#     """

#     xs_permuted = xs[..., feature_permute_idx]  # [T, N, F]

#     T, N, _ = xs_permuted.shape
#     perm_T = torch.randperm(T)  # First permute the features in time
#     xs_permuted = xs_permuted[perm_T, ...]

#     # Then permute the features within each ensemble member
#     # Shuffle across N dimension, but do so differently for each time step T
#     indices = torch.argsort(torch.rand((T, N)), dim=1).unsqueeze(-1).repeat(1, 1, len(feature_permute_idx))
#     result = torch.gather(xs_permuted, dim=1, index=indices)

#     # Replace features with permuted features
#     xs[..., feature_permute_idx] = result
#     return xs


# def rm_edges(data: List[torch_geometric.data.Data]) -> None:
#     """Remove all edges from the graphs in the list.

#     Args:
#         data (List[torch_geometric.data.Data]): List of graphs
#     """
#     for graph in data:
#         graph.edge_index = torch.empty(2, 0, dtype=torch.long)
#         graph.edge_attr = torch.empty(0)


# def summary_statistics(dataframes: defaultdict) -> defaultdict:
#     """
#     Calculate summary statistics for each feature dataframe in the given dictionary.
#     The dictionary can contain multiple tuples which contain the dataframe and the target values.
#     Also the dict can contain a dataframe with the stations, which will be returned unaltered.

#     Args:
#         dataframes (defaultdict): A dictionary containing the dataframes to calculate summary statistics for.

#     Returns:
#         defaultdict: A dictionary containing the updated dataframes with summary statistics.

#     """
#     only_mean = ["model_orography", "station_altitude", "station_latitude", "station_longitude"]
#     for key, df in dataframes.items():
#         if key == "stations":
#             continue
#         print(f"[INFO] Calculating summary statistics for {key}")
#         y = df[1]
#         df = df[0]

#         rest = [col for col in df.columns if col not in only_mean]

#         mean_agg = df.groupby(["time", "station_id"])[only_mean].agg("mean")
#         rest_agg = (
#             df.groupby(["time", "station_id"])[rest].agg(["mean", "std"]).drop(columns=["number", "station_id", "time"])
#         )
#         rest_agg.columns = ["_".join(col).strip() for col in rest_agg.columns.values]
#         df = pd.concat([mean_agg, rest_agg], axis=1).reset_index()
#         df["number"] = 0
#         dataframes[key] = (df, y)
#     return dataframes


# def get_mask(
#     dist_matrix_sliced: np.array, method: str = "max_dist", k: int = 3, max_dist: int = 50, nearest_k_mode: str = "in"
# ) -> np.array:
#     """
#     Generate mask which specifies which edges to include in the graph
#     :param dist_matrix_sliced: distance matrix with only the reporting stations
#     :param method: method to compute included edges. max_dist includes all edges wich are shorter than max_dist km,
#     knn includes the k nearest edges for each station. So the out_degree of each station is k,
#     the in_degree can vary.
#     :param k: number of connections per node
#     :param max_dist: maximum length of edges
#     :param nearest_k_mode: "in" or "out". If "in" the every node has k nodes passing information to it,
#     if "out" every node passes information to k nodes.
#     :return: return boolean mask of which edges to include
#     :rtype: np.array
#     """
#     mask = None
#     if method == "max_dist":
#         mask = (dist_matrix_sliced <= max_dist) & (dist_matrix_sliced != 0)
#     elif method == "knn":
#         k = k + 1
#         nearest_indices = np.argsort(dist_matrix_sliced, axis=1)[:, 1:k]
#         # Create an empty boolean array with the same shape as distances
#         nearest_k = np.zeros_like(dist_matrix_sliced, dtype=bool)
#         # Set the corresponding indices in the nearest_k array to True
#         row_indices = np.arange(dist_matrix_sliced.shape[0])[:, np.newaxis]
#         nearest_k[row_indices, nearest_indices] = True
#         if nearest_k_mode == "in":
#             mask = nearest_k.T
#         elif nearest_k_mode == "out":
#             mask = nearest_k

#     return mask


# def generate_layers(n_nodes, n_layers) -> np.array:
#     """
#     Generate bidirectional connections between nodes x_{s,n}, where s has the same value.

#     Args:
#         n_nodes (int): Number of stations (122).
#         n_layers (int): Number of ensemble members (11 or 51).

#     Returns:
#         np.array: Bidirectional connections between layers of nodes.

#     """
#     all_layers = []
#     start_i = 0
#     start_j = n_nodes
#     for i in range(n_layers):
#         layer_i = np.arange(start_i, start_i + n_nodes)
#         start_j = start_i + n_nodes
#         for j in range(i + 1, n_layers):
#             layer_array = np.empty((2, n_nodes), dtype=int)
#             layer_array[0] = layer_i
#             layer_array[1] = np.arange(start_j, start_j + n_nodes)
#             all_layers.append(layer_array)
#             start_j += n_nodes
#         start_i += n_nodes
#     connections = np.hstack(all_layers)
#     connections_bidirectional = np.hstack([connections, np.flip(connections, axis=0)])
#     return connections_bidirectional


# def create_multigraph(df, df_target, distances, max_dist):
#     """Create a multigraph from the input data.

#     Args:
#         df (pd.DataFrame)): feature dataframe
#         df_target (pd.DataFrame): target dataframe
#         distances (np.ndarray): distance matrix
#         max_dist (float): maximum distance for edges

#     Returns:
#         List[torch_geometric.data.Data]: List of graphs with features and targets
#     """
#     n_nodes = len(df.station_id.unique())  # number of nodes
#     n_fc = len(df.number.unique())  # number of ensembe members
#     df = df.drop(columns=["number"])

#     # Create set of edges ######################################################################
#     mask = get_mask(distances, max_dist=max_dist)
#     edges = np.argwhere(mask)
#     edge_index = edges.T  # (2, num_edges) holds indices of connected nodes for one level (forecast) of the multigraph

#     # Create edge features
#     edge_attr = distances[edges[:, 0], edges[:, 1]]
#     edge_attr = edge_attr.reshape(-1, 1)
#     max_len = np.max(edge_attr)
#     standardized_edge_attr = edge_attr / max_len

#     # Repeat edge_attr for all levels of the multigraph
#     full_edge_attr = np.repeat(standardized_edge_attr, n_fc, axis=1).T.reshape(-1, 1)

#     # Get all other Levels of the Multigraph
#     values_to_add = np.arange(n_fc) * (n_nodes)
#     stacked = np.repeat(edge_index[np.newaxis, ...], n_fc, axis=0) + values_to_add[:, np.newaxis, np.newaxis]
#     full_edge_index = stacked.transpose(1, 0, 2).reshape(2, -1)

#     # Add connections between levels
#     if n_fc > 1:
#         full_edge_index = np.hstack([full_edge_index, generate_layers(n_nodes, n_layers=n_fc)])

#     # Add 0 for each remaining edge attribute
#     connections = (
#         np.ones((full_edge_index.shape[1] - full_edge_attr.shape[0], 1)) * 0.01
#     )  # fill connections with small value
#     full_edge_attr = np.vstack([full_edge_attr, connections])
#     full_edge_attr = torch.tensor(full_edge_attr, dtype=torch.float32)

#     # Create node features ######################################################################
#     graphs = []
#     for time in df.time.unique():
#         day = df[df.time == time]  # get all forecasts for one day
#         day = day.drop(columns=["time"])
#         x = torch.tensor(day.to_numpy(dtype=np.float32))
#         assert x.shape[0] == n_nodes * n_fc

#         day_target = df_target[df_target.time == time]
#         # ! TODO This should surely be done somerwhere else
#         #y = torch.tensor(day_target["tp6"].to_numpy(dtype=np.float32)) - 273.15  # ! convert to celsius
#         y = torch.tensor(day_target["tp6"].to_numpy(dtype=np.float32))
#         assert y.shape[0] == n_nodes, f"y.shape[0] = {y.shape[0]}, n_nodes = {n_nodes}, time = {time}, {day_target}"
#         pyg_data = Data(
#             x=x,
#             edge_index=torch.tensor(full_edge_index, dtype=torch.long),
#             edge_attr=full_edge_attr,
#             y=y,
#             timestamp=time,
#             n_idx=torch.arange(n_nodes).repeat(n_fc),
#         )
#         graphs.append(pyg_data)
#     return graphs

# # # data.py
# # import os
# # import numpy as np
# # import pandas as pd
# # import torch

# # import torch_geometric
# # from torch_geometric.data import Data
# # from sklearn.preprocessing import StandardScaler
# # from collections import defaultdict
# # from typing import DefaultDict, Tuple, List

# # #####################################################
# # # Minimal Zarr Loader (optional)
# # #####################################################

# # import xarray
# # class ZarrLoader:
# #     """
# #     A class for loading data from Zarr files.

# #     Args:
# #         data_path (str): The path to the data directory.

# #     Attributes:
# #         data_path (str): The path to the data directory.
# #         leadtime (pd.Timedelta): The lead time for the forecasts.
# #         countries (List[str]): The list of countries to load data for.
# #         features (List[str]): The list of features to load.

# #     Methods:
# #         get_stations(arr: xarray.Dataset) -> pd.DataFrame:
# #             Get the stations information from the dataset.

# #         load_data(leadtime: str = "24h", countries: Union[str, List[str]] = "all",
# #         features: Union[str, List[str]] = "all")
# #         -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset, xarray.Dataset]:
# #             Load the data from Zarr files.

# #         validate_stations() -> bool:
# #             Validate if the station IDs match between forecasts and reforecasts.
# #     """

# #     def __init__(self, data_path: str) -> None:
# #         self.data_path = data_path

# #     def get_stations(self, arr: xarray.Dataset) -> pd.DataFrame:
# #         """
# #         Get the stations information from the dataset.

# #         Args:
# #             arr (xarray.Dataset): The dataset containing station information.

# #         Returns:
# #             pd.DataFrame: The dataframe containing station information.
# #         """
# #         stations = pd.DataFrame(
# #             {
# #                 "station_id": arr.station_id.values,
# #                 "lat": arr.station_latitude.values,
# #                 "lon": arr.station_longitude.values,
# #                 "altitude": arr.station_altitude.values,
# #                 "name": arr.station_name.values,
# #             }
# #         )
# #         stations = stations.sort_values("station_id").reset_index(drop=True)
# #         return stations

# #     def load_data(
# #         self, leadtime: str = "24h", countries: Union[str, List[str]] = "all", features: Union[str, List[str]] = "all"
# #     ) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset, xarray.Dataset]:
# #         """
# #         Load data for the specified lead time, countries, and features.

# #         Args:
# #             leadtime (str): The lead time for the forecasts and reforecasts. Default is "24h".
# #             countries (Union[str, List[str]]): The countries for which to load the data. Default is "all".
# #             features (Union[str, List[str]]): The features to load. Default is "all".

# #         Returns:
# #             Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset, xarray.Dataset]:
# #             A tuple containing the following datasets:
# #                 - df_f: The forecasts dataset.
# #                 - df_f_target: The targets for the forecasts dataset.
# #                 - df_rf: The reforecasts dataset.
# #                 - df_rf_target: The targets for the reforecasts dataset.
# #         """
# #         self.leadtime = pd.Timedelta(leadtime)

# #         if countries == "all":
# #             print("[INFO] Loading data for all countries")
# #             self.countries = ["austria", "belgium", "france", "germany", "netherlands"]
# #         elif isinstance(countries, list):
# #             print(f"[INFO] Loading data for {countries}")
# #             self.countries = countries
# #         else:
# #             raise ValueError("countries must be a list of strings or 'all'")

# #         if features == "all":
# #             print("[INFO] Loading all features")
# #             self.features = ["number"] + [
# #                 "station_id",
# #                 "time",
# #                 "cape",
# #                 "model_orography",
# #                 "sd",
# #                 "station_altitude",
# #                 "station_latitude",
# #                 "station_longitude",
# #                 "stl1",
# #                 "swvl1",
# #                 "t2m",
# #                 "tcc",
# #                 "tcw",
# #                 "tcwv",
# #                 "u10",
# #                 "u100",
# #                 "v10",
# #                 "v100",
# #                 "vis",
# #                 "cp6",
# #                 "mn2t6",
# #                 "mx2t6",
# #                 "p10fg6",
# #                 "slhf6",
# #                 "sshf6",
# #                 "ssr6",
# #                 "ssrd6",
# #                 "str6",
# #                 "strd6",
# #                 "tp6",
# #                 "z",
# #                 "q",
# #                 "u",
# #                 "v",
# #                 "t",
# #             ]
# #         elif isinstance(features, list):
# #             print(f"[INFO] Loading features: {features}")
# #             self.features = ["number"] + features
# #         else:
# #             raise ValueError("features must be a list of strings or 'all'")

# #         # Load Data from Zarr ####
# #         forecasts_all_countries = []
# #         reforecasts_all_countries = []

# #         targets_f_all_countries = []
# #         targets_rf_all_countries = []
# #         for country in self.countries:
# #             print(f"[INFO] Loading data for {country}")
# #             # Forecasts
# #             f_surface_xr = xarray.open_zarr(f"{self.data_path}/stations_ensemble_forecasts_surface_{country}.zarr")
# #             f_surface_pp_xr = xarray.open_zarr(
# #                 f"{self.data_path}/stations_ensemble_forecasts_surface_postprocessed_{country}.zarr"
# #             )
# #             f_pressure_500_xr = xarray.open_zarr(
# #                 f"{self.data_path}/stations_ensemble_forecasts_pressure_500_{country}.zarr"
# #             )
# #             f_pressure_700_xr = xarray.open_zarr(
# #                 f"{self.data_path}/stations_ensemble_forecasts_pressure_700_{country}.zarr"
# #             )
# #             f_pressure_850_xr = xarray.open_zarr(
# #                 f"{self.data_path}/stations_ensemble_forecasts_pressure_850_{country}.zarr"
# #             )
# #             f_obs_xr = xarray.open_zarr(f"{self.data_path}/stations_forecasts_observations_surface_postprocessed_{country}.zarr")
# #             forecasts = [f_surface_xr, f_surface_pp_xr, f_pressure_500_xr, f_pressure_700_xr, f_pressure_850_xr]

# #             # Reforecasts
# #             rf_surface_xr = xarray.open_zarr(f"{self.data_path}/stations_ensemble_reforecasts_surface_{country}.zarr")
# #             rf_surface_pp_xr = xarray.open_zarr(
# #                 f"{self.data_path}/stations_ensemble_reforecasts_surface_postprocessed_{country}.zarr"
# #             )
# #             rf_pressure_500_xr = xarray.open_zarr(
# #                 f"{self.data_path}/stations_ensemble_reforecasts_pressure_500_{country}.zarr"
# #             )
# #             rf_pressure_700_xr = xarray.open_zarr(
# #                 f"{self.data_path}/stations_ensemble_reforecasts_pressure_700_{country}.zarr"
# #             )
# #             rf_pressure_850_xr = xarray.open_zarr(
# #                 f"{self.data_path}/stations_ensemble_reforecasts_pressure_850_{country}.zarr"
# #             )
# #             rf_obs_xr = xarray.open_zarr(f"{self.data_path}/stations_reforecasts_observations_surface_postprocessed_{country}.zarr")
# #             reforecasts = [rf_surface_xr, rf_surface_pp_xr, rf_pressure_500_xr, rf_pressure_700_xr, rf_pressure_850_xr]

# #             forecasts = [forecast.drop_vars("valid_time").squeeze(drop=True) for forecast in forecasts]
# #             reforecasts = [reforecast.drop_vars("valid_time").squeeze(drop=True) for reforecast in reforecasts]

# #             forecasts = xarray.merge(forecasts).sel(step=self.leadtime)
# #             reforecasts = xarray.merge(reforecasts).sel(step=self.leadtime)

# #             forecasts_all_countries.append(forecasts)
# #             reforecasts_all_countries.append(reforecasts)

# #             targets_f = f_obs_xr.squeeze(drop=True).sel(step=self.leadtime)
# #             targets_rf = rf_obs_xr.squeeze(drop=True).sel(step=self.leadtime)

# #             targets_f_all_countries.append(targets_f)
# #             targets_rf_all_countries.append(targets_rf)

# #         forecasts = xarray.concat(forecasts_all_countries, dim="station_id")
# #         reforecasts = xarray.concat(reforecasts_all_countries, dim="station_id")

# #         targets_f = xarray.concat(targets_f_all_countries, dim="station_id")
# #         targets_rf = xarray.concat(targets_rf_all_countries, dim="station_id")

# #         forecasts = forecasts.drop_vars(
# #             ["model_altitude", "model_land_usage", "model_latitude", "model_longitude", "station_land_usage", "step"]
# #         )
# #         reforecasts = reforecasts.drop_vars(
# #             ["model_altitude", "model_land_usage", "model_latitude", "model_longitude", "station_land_usage", "step"]
# #         )
# #         print(
# #             f"[INFO] Data loaded successfully. Forecasts shape:\
# #             {forecasts.tp6.shape}, Reforecasts shape: {reforecasts.tp6.shape}"
# #         )
# #         # Extract Stations ####
# #         self.stations_f = self.get_stations(forecasts)
# #         self.stations_rf = self.get_stations(reforecasts)

# #         # Turn into pandas Dataframe ####
# #         df_f = (
# #             forecasts.to_dataframe()
# #             .reorder_levels(["time", "number", "station_id"])
# #             .sort_index(level=["time", "number", "station_id"])
# #             .reset_index()
# #         )
# #         df_f_target = (
# #             targets_f.tp6.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name", "step"])
# #             .to_dataframe()
# #             .reorder_levels(["time", "station_id"])
# #             .sort_index(level=["time", "station_id"])
# #             .reset_index()
# #         )

# #         df_rf = reforecasts.to_dataframe().reset_index()
# #         df_rf_target = (
# #             targets_rf.tp6.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name", "step"])
# #             .to_dataframe()
# #             .reset_index()
# #         )

# #         df_rf["time"] = df_rf["time"] - df_rf["year"].apply(lambda x: pd.Timedelta((21 - x) * 365, unit="day"))
# #         df_rf_target["time"] = df_rf_target["time"] - df_rf_target["year"].apply(
# #             lambda x: pd.Timedelta((21 - x) * 365, unit="day")  # ! 21 or 20 years of reforecasts
# #         )

# #         df_rf = df_rf.drop(columns=["year"]).reindex(columns=df_f.columns).sort_values(["time", "number", "station_id"])
# #         df_rf_target = (
# #             df_rf_target.drop(columns=["year"]).reindex(columns=df_f_target.columns).sort_values(["time", "station_id"])
# #         )

# #         # Turn Station IDs into a Station Index starting from 0
# #         station_ids = df_f.station_id.unique()
# #         id_to_index = {station_id: i for i, station_id in enumerate(station_ids)}

# #         df_f["station_id"] = df_f["station_id"].apply(lambda x: id_to_index[x])
# #         df_f_target["station_id"] = df_f_target["station_id"].apply(lambda x: id_to_index[x])
# #         df_rf["station_id"] = df_rf["station_id"].apply(lambda x: id_to_index[x])
# #         df_rf_target["station_id"] = df_rf_target["station_id"].apply(lambda x: id_to_index[x])

# #         # Transform precipitation
# #         # Transform units to mm and apply log(x+0.01) transformation
# #         df_f_target["tp6"] = np.log(df_f_target["tp6"].clip(lower = 0)*1000 + 0.01)
# #         df_rf_target["tp6"] = np.log(df_rf_target["tp6"].clip(lower = 0)*1000 + 0.01)

# #         df_f["tp6"] = np.log(df_f["tp6"].clip(lower = 0)*1000 + 0.01)
# #         df_rf["tp6"] = np.log(df_rf["tp6"].clip(lower = 0)*1000 + 0.01)

# #         # Cut features ####
# #         df_f = df_f[self.features]
# #         df_rf = df_rf[self.features]

# #         return df_f, df_f_target, df_rf, df_rf_target

# #     def validate_stations(self):
# #         return (self.stations_f.station_id == self.stations_rf.station_id).all()

# # #####################################################
# # # Main data loading function
# # #####################################################

# # def load_dataframes(
# #     mode: str,
# #     leadtime: str,
# # ) -> DefaultDict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
# #     """
# #     Load dataframes for training or evaluation from pickled files if they exist,
# #     otherwise load from Zarr (ZarrLoader) and save them.

# #     Returns a dict with:
# #       "train": (train_features, train_targets)
# #       "test_rf": (test_reforecast_features, test_reforecast_targets)
# #       "test_f": (test_forecast_features, test_forecast_targets)
# #       "stations": stations_df
# #     """

# #     data_dir = f"data/dataframes_{leadtime}/final_train"
# #     #data_dir = f"/home/groups/ai/buelte/precip/Singapur-Trip-25/data/dataframes_{leadtime}"
# #     os.makedirs(data_dir, exist_ok=True)

# #     # File paths
# #     TRAIN_RF_PATH = os.path.join(data_dir, "train_rf_final.pkl")
# #     TRAIN_RF_TARGET_PATH = os.path.join(data_dir, "train_rf_target_final.pkl")
# #     TEST_RF_PATH = os.path.join(data_dir, "valid_rf_final.pkl")
# #     TEST_RF_TARGET_PATH = os.path.join(data_dir, "valid_rf_target_final.pkl")
# #     TEST_F_PATH = os.path.join(data_dir, "test_f_final.pkl")
# #     TEST_F_TARGET_PATH = os.path.join(data_dir, "test_f_target_final.pkl")
# #     STATIONS_PATH = os.path.join(data_dir, "stations.pkl")

# #     res = defaultdict(lambda: None)
# #     # Check if pickles exist
# #     if (
# #         os.path.isfile(TRAIN_RF_PATH)
# #         and os.path.isfile(TRAIN_RF_TARGET_PATH)
# #         and os.path.isfile(TEST_RF_PATH)
# #         and os.path.isfile(TEST_RF_TARGET_PATH)
# #         and os.path.isfile(TEST_F_PATH)
# #         and os.path.isfile(TEST_F_TARGET_PATH)
# #         and os.path.isfile(STATIONS_PATH)
# #     ):
# #         print("[INFO] Dataframes found locally. Loading from pickles...")
# #         train_rf = pd.read_pickle(TRAIN_RF_PATH)
# #         train_rf_target = pd.read_pickle(TRAIN_RF_TARGET_PATH)
# #         test_rf = pd.read_pickle(TEST_RF_PATH)
# #         test_rf_target = pd.read_pickle(TEST_RF_TARGET_PATH)
# #         test_f = pd.read_pickle(TEST_F_PATH)
# #         test_f_target = pd.read_pickle(TEST_F_TARGET_PATH)
# #         stations_df = pd.read_pickle(STATIONS_PATH)

# #     else:
# #         print("[INFO] Pickles not found. Attempting to load from Zarr...")
# #         loader = ZarrLoader("/home/groups/ai/buelte/precip/Singapur-Trip-25/data/EUPPBench-stations")
# #         # This function would produce df_f, df_f_target, df_rf, df_rf_target
# #         df_f, df_f_target, df_rf, df_rf_target = loader.load_data(leadtime=leadtime)

# #         # Example station DataFrame placeholder
# #         # A real environment might produce this from the Zarr files
# #         stations_df = pd.DataFrame({
# #             "station_id": df_f["station_id"].unique(),
# #             "lat": np.random.rand(len(df_f["station_id"].unique())) * 10 + 40,
# #             "lon": np.random.rand(len(df_f["station_id"].unique())) * 5 - 5
# #         })

# #         # Basic time splits for demonstration
# #         train_cutoff = pd.Timestamp("2014-01-01")
# #         train_rf = df_rf[df_rf["time"] < train_cutoff]
# #         train_rf_target = df_rf_target[df_rf_target["time"] < train_cutoff]

# #         test_rf = df_rf[df_rf["time"] >= train_cutoff]
# #         test_rf_target = df_rf_target[df_rf_target["time"] >= train_cutoff]

# #         test_f = df_f
# #         test_f_target = df_f_target

# #         # Save to pickle
# #         train_rf.to_pickle(TRAIN_RF_PATH)
# #         train_rf_target.to_pickle(TRAIN_RF_TARGET_PATH)
# #         test_rf.to_pickle(TEST_RF_PATH)
# #         test_rf_target.to_pickle(TEST_RF_TARGET_PATH)
# #         test_f.to_pickle(TEST_F_PATH)
# #         test_f_target.to_pickle(TEST_F_TARGET_PATH)
# #         stations_df.to_pickle(STATIONS_PATH)

# #     if mode == "eval":
# #         # In many setups, "train" and "eval" might load the same training data
# #         # For minimal code, we do:
# #         pass

# #     res["train"] = (train_rf, train_rf_target)
# #     res["test_rf"] = (test_rf, test_rf_target)
# #     res["test_f"] = (test_f, test_f_target)
# #     res["stations"] = stations_df
# #     return res


# # def load_distances(stations: pd.DataFrame) -> np.ndarray:
# #     """
# #     Load or compute a distance matrix for the station set.
# #     We'll store the result in data/distances.npy if not present.
# #     """
# #     distance_file = "data/distances.npy"
# #     if os.path.isfile(distance_file):
# #         print("[INFO] Loading distances from file:", distance_file)
# #         dist_mat = np.load(distance_file)
# #         return dist_mat
# #     else:
# #         print("[INFO] Computing station distance matrix...")
# #         dist_mat = compute_dist_matrix(stations)
# #         os.makedirs("data", exist_ok=True)
# #         np.save(distance_file, dist_mat)
# #         return dist_mat


# # def compute_dist_matrix(df: pd.DataFrame) -> np.ndarray:
# #     """
# #     Minimal approach: Euclidean distance for demonstration.
# #     (Or you could use geopy if lat/lon are real coordinates.)
# #     """
# #     coords = df[["lat", "lon"]].to_numpy()
# #     N = coords.shape[0]
# #     dist_mat = np.zeros((N, N), dtype=np.float32)
# #     for i in range(N):
# #         for j in range(i + 1, N):
# #             dist = np.linalg.norm(coords[i] - coords[j])
# #             dist_mat[i, j] = dist
# #             dist_mat[j, i] = dist
# #     return dist_mat


# # def build_edge_index_and_attr(dist_mat: np.ndarray, max_dist: float):
# #     """
# #     Build edge_index and edge_attr from a distance matrix by selecting edges within 'max_dist'.
# #     """
# #     D = dist_mat.copy()
# #     np.fill_diagonal(D, np.inf)
# #     mask = (D <= max_dist)
# #     row, col = np.where(mask)
# #     valid_distances = D[row, col]

# #     edge_index = torch.tensor([row, col], dtype=torch.long)
# #     edge_attr = torch.tensor(valid_distances, dtype=torch.float32).unsqueeze(1)
# #     return edge_index, edge_attr


# # def create_graphs_per_time(df_features, df_targets, dist_mat, max_dist):
# #     """
# #     Creates one PyG Data object per unique time stamp.
# #     Minimal approach:
# #         x = first row's features per station
# #         ensemble = all rows [N, E, #features]
# #         y = [N]
# #     """
# #     edge_index, edge_attr = build_edge_index_and_attr(dist_mat, max_dist)

# #     graphs = []
# #     for t in df_features["time"].unique():
# #         sub_feat = df_features[df_features["time"] == t].sort_values(["station_id", "number"])
# #         sub_targ = df_targets[df_targets["time"] == t].sort_values("station_id")

# #         stations = sub_feat["station_id"].unique()
# #         E = sub_feat["number"].nunique()
# #         N = len(stations)

# #         # We'll pick numeric columns as ensemble features
# #         exclude_cols = ["station_id", "time", "number"]
# #         feat_cols = [c for c in sub_feat.columns if c not in exclude_cols]

# #         # shape => [N, E, feats]
# #         feats_array = sub_feat[feat_cols].to_numpy(dtype=np.float32).reshape(N, E, -1)
# #         x_extra = feats_array[:, 0, :]  # first row per station
# #         ensemble_tensor = torch.tensor(feats_array, dtype=torch.float32)

# #         y_vals = sub_targ["tp6"].to_numpy(dtype=np.float32)
# #         y_tensor = torch.tensor(y_vals)

# #         data = Data(
# #             x=torch.tensor(x_extra, dtype=torch.float32),
# #             ensemble=ensemble_tensor,
# #             edge_index=edge_index,
# #             edge_attr=edge_attr,
# #             y=y_tensor
# #         )
# #         graphs.append(data)
# #     return graphs


# # def normalize_features_and_create_graphs(
# #     training_data: Tuple[pd.DataFrame, pd.DataFrame],
# #     valid_test_data: List[Tuple[pd.DataFrame, pd.DataFrame]],
# #     mat: np.ndarray,
# #     max_dist: float,
# #     new_gnn=False
# # ):
# #     """
# #     Fit a StandardScaler on training features, apply to test data, then create PyG Data objects.
# #     """
# #     (train_df, train_target) = training_data

# #     # Identify numeric columns
# #     exclude_cols = ["station_id", "time", "number"]
# #     numeric_cols = [c for c in train_df.columns if c not in exclude_cols]

# #     scaler = StandardScaler()
# #     train_df = train_df.copy()
# #     train_df.loc[:, numeric_cols] = scaler.fit_transform(train_df[numeric_cols])

# #     vt_dataframes = []
# #     for (dfv, dfv_target) in valid_test_data:
# #         dfv = dfv.copy()
# #         dfv.loc[:, numeric_cols] = scaler.transform(dfv[numeric_cols])
# #         vt_dataframes.append((dfv, dfv_target))

# #     # Build training graphs
# #     graphs_train = create_graphs_per_time(train_df, train_target, mat, max_dist)

# #     # Build val/test graphs
# #     vt_graphs = []
# #     for (dfv, dfv_target) in vt_dataframes:
# #         g = create_graphs_per_time(dfv, dfv_target, mat, max_dist)
# #         vt_graphs.append(g)

# #     return graphs_train, vt_graphs


# # def split_graph(graph, new_gnn=False):
# #     """
# #     Example function to split a graph containing [N, E, feats] into multiple subgraphs.
# #     For demonstration only, keep or adapt as needed.

# #     If new_gnn=True, we might split the 'ensemble' dimension into chunks.
# #     """
# #     if not new_gnn:
# #         # Return the single graph as-is
# #         return [graph]
# #     # Otherwise, pretend we break it into 5 subgraphs
# #     # if the ensemble dimension is E=50, each chunk is 10 members, etc.
# #     ensemble_tensor = graph.ensemble
# #     _, E, F = ensemble_tensor.shape
# #     chunk_size = E // 5 if E >= 5 else 1

# #     subgraphs = []
# #     start = 0
# #     for _ in range(5):
# #         end = min(start + chunk_size, E)
# #         ens_chunk = ensemble_tensor[:, start:end, :]
# #         new_graph = graph.clone()
# #         new_graph.ensemble = ens_chunk
# #         subgraphs.append(new_graph)
# #         start = end
# #     return subgraphs

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
from typing import DefaultDict, Tuple, List, Union
import geopy.distance
import xarray
from sklearn.preprocessing import StandardScaler


############################################################################
# 1. ZarrLoader for EUPPBench
############################################################################

class ZarrLoader:
    """
    Loads EUPPBench data from Zarr archives. Provides:
      - self.load_data(...) => (df_f, df_f_target, df_rf, df_rf_target)
      - self.stations_f, self.stations_rf => station info for forecasts/reforecasts
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Directory containing the Zarr archives, e.g.,
                             ".../EUPPBench-stations"
        """
        self.data_path = data_path

    def get_stations(self, arr: xarray.Dataset) -> pd.DataFrame:
        """
        Extract station info from a Dataset (arr).
        """
        stations = pd.DataFrame({
            "station_id": arr.station_id.values,
            "lat": arr.station_latitude.values,
            "lon": arr.station_longitude.values,
            "altitude": arr.station_altitude.values,
            "name": arr.station_name.values,
        })
        return stations.sort_values("station_id").reset_index(drop=True)

    def load_data(
        self,
        leadtime: str = "24h",
        countries: Union[str, List[str]] = "all",
        features: Union[str, List[str]] = "all"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads EUPPBench forecasts & reforecasts from Zarr, merges them,
        and returns four DataFrames: df_f, df_f_target, df_rf, df_rf_target.

        Args:
            leadtime (str): e.g. "24h", "72h". Applied as a pd.Timedelta to each dataset.
            countries: "all" or list of country names, e.g. ["germany", "france"]
            features: "all" or list of features to load.

        Returns:
            (df_f, df_f_target, df_rf, df_rf_target) as DataFrames
        """
        self.leadtime = pd.Timedelta(leadtime)

        # Decide which countries
        if countries == "all":
            print("[INFO] Loading data for all countries")
            self.countries = ["austria", "belgium", "france", "germany", "netherlands"]
        elif isinstance(countries, list):
            print(f"[INFO] Loading data for: {countries}")
            self.countries = countries
        else:
            raise ValueError("countries must be 'all' or a list of strings")

        # Decide which features
        if features == "all":
            print("[INFO] Loading all default features.")
            self.features = ["number"] + [
                "station_id", "time", "cape", "model_orography", "sd",
                "station_altitude", "station_latitude", "station_longitude",
                "stl1", "swvl1", "t2m", "tcc", "tcw", "tcwv", "u10", "u100",
                "v10", "v100", "vis", "cp6", "mn2t6", "mx2t6", "p10fg6",
                "slhf6", "sshf6", "ssr6", "ssrd6", "str6", "strd6",
                "tp6", "z", "q", "u", "v", "t",
            ]
            # Also add cyclical features
            self.features += ["cos_doy", "sin_doy"]
        elif isinstance(features, list):
            print(f"[INFO] Loading specified features: {features}")
            self.features = ["number"] + features
            # If you want cyclical columns even with user-specified features:
            # ensure they are appended
            if "cos_doy" not in self.features:
                self.features.append("cos_doy")
            if "sin_doy" not in self.features:
                self.features.append("sin_doy")
        else:
            raise ValueError("features must be 'all' or a list of strings")

        # Prepare containers
        forecasts_all, reforecasts_all = [], []
        targets_f_all, targets_rf_all = [], []

        # Load from Zarr for each country
        for country in self.countries:
            print(f"[INFO] Loading data for {country}")
            # Forecasts (surface, postprocessed, pressure_500,700,850)
            fc_surface   = xarray.open_zarr(f"{self.data_path}/stations_ensemble_forecasts_surface_{country}.zarr")
            fc_surface_pp= xarray.open_zarr(f"{self.data_path}/stations_ensemble_forecasts_surface_postprocessed_{country}.zarr")
            fc_p500      = xarray.open_zarr(f"{self.data_path}/stations_ensemble_forecasts_pressure_500_{country}.zarr")
            fc_p700      = xarray.open_zarr(f"{self.data_path}/stations_ensemble_forecasts_pressure_700_{country}.zarr")
            fc_p850      = xarray.open_zarr(f"{self.data_path}/stations_ensemble_forecasts_pressure_850_{country}.zarr")
            obs_f_xr     = xarray.open_zarr(f"{self.data_path}/stations_forecasts_observations_surface_postprocessed_{country}.zarr")

            # Reforecasts
            rf_surface   = xarray.open_zarr(f"{self.data_path}/stations_ensemble_reforecasts_surface_{country}.zarr")
            rf_surface_pp= xarray.open_zarr(f"{self.data_path}/stations_ensemble_reforecasts_surface_postprocessed_{country}.zarr")
            rf_p500      = xarray.open_zarr(f"{self.data_path}/stations_ensemble_reforecasts_pressure_500_{country}.zarr")
            rf_p700      = xarray.open_zarr(f"{self.data_path}/stations_ensemble_reforecasts_pressure_700_{country}.zarr")
            rf_p850      = xarray.open_zarr(f"{self.data_path}/stations_ensemble_reforecasts_pressure_850_{country}.zarr")
            obs_rf_xr    = xarray.open_zarr(f"{self.data_path}/stations_reforecasts_observations_surface_postprocessed_{country}.zarr")

            # Remove "valid_time" & unify
            fc_list = [xr.drop_vars("valid_time").squeeze(drop=True) for xr in [fc_surface, fc_surface_pp, fc_p500, fc_p700, fc_p850]]
            rf_list = [xr.drop_vars("valid_time").squeeze(drop=True) for xr in [rf_surface, rf_surface_pp, rf_p500, rf_p700, rf_p850]]

            fc_merged = xarray.merge(fc_list).sel(step=self.leadtime)
            rf_merged = xarray.merge(rf_list).sel(step=self.leadtime)

            forecasts_all.append(fc_merged)
            reforecasts_all.append(rf_merged)

            # Observations
            obs_f  = obs_f_xr.squeeze(drop=True).sel(step=self.leadtime)
            obs_rf = obs_rf_xr.squeeze(drop=True).sel(step=self.leadtime)
            targets_f_all.append(obs_f)
            targets_rf_all.append(obs_rf)

        # Concat all countries
        forecasts  = xarray.concat(forecasts_all,  dim="station_id")
        reforecasts= xarray.concat(reforecasts_all, dim="station_id")
        f_targets  = xarray.concat(targets_f_all,   dim="station_id")
        rf_targets = xarray.concat(targets_rf_all,  dim="station_id")

        # Drop unneeded coords
        drop_coords = [
            "model_altitude", "model_land_usage", "model_latitude",
            "model_longitude", "station_land_usage", "step"
        ]
        forecasts   = forecasts.drop_vars(drop_coords)
        reforecasts = reforecasts.drop_vars(drop_coords)

        print(f"[INFO] Loaded Forecasts shape: {forecasts.tp6.shape}, Reforecasts shape: {reforecasts.tp6.shape}")

        # Station info
        self.stations_f = self.get_stations(forecasts)
        self.stations_rf= self.get_stations(reforecasts)

        # Convert to DataFrame
        df_f = (forecasts.to_dataframe()
                .reorder_levels(["time","number","station_id"])
                .sort_index(level=["time","number","station_id"])
                .reset_index())
        df_f_target = (f_targets.tp6.drop_vars([
            "altitude","land_usage","latitude","longitude","station_name","step"
        ]).to_dataframe()
          .reorder_levels(["time","station_id"])
          .sort_index(level=["time","station_id"])
          .reset_index())

        df_rf = reforecasts.to_dataframe().reset_index()
        df_rf_target = (rf_targets.tp6.drop_vars([
            "altitude","land_usage","latitude","longitude","station_name","step"
        ]).to_dataframe()
          .reset_index())

        # Reforecasts: shift 'time' by (21 - year) * 365 days
        df_rf["time"] = df_rf["time"] - df_rf["year"].apply(lambda x: pd.Timedelta((21 - x)*365, unit="day"))
        df_rf_target["time"] = df_rf_target["time"] - df_rf_target["year"].apply(
            lambda x: pd.Timedelta((21 - x)*365, unit="day")
        )
        df_rf = df_rf.drop(columns=["year"])
        df_rf_target = df_rf_target.drop(columns=["year"])

        # Sort reforecast columns to match forecast columns
        df_rf = (df_rf.reindex(columns=df_f.columns)
                       .sort_values(["time","number","station_id"]))
        df_rf_target = (df_rf_target.reindex(columns=df_f_target.columns)
                                   .sort_values(["time","station_id"]))

        # Map station_id => [0..N-1]
        station_ids = df_f.station_id.unique()
        id_to_idx   = {sid: i for i,sid in enumerate(station_ids)}

        df_f["station_id"]        = df_f["station_id"].apply(lambda x: id_to_idx[x])
        df_f_target["station_id"] = df_f_target["station_id"].apply(lambda x: id_to_idx[x])
        df_rf["station_id"]       = df_rf["station_id"].apply(lambda x: id_to_idx[x])
        df_rf_target["station_id"]= df_rf_target["station_id"].apply(lambda x: id_to_idx[x])

        # Convert precip to mm + log transform
        for d in [df_f, df_f_target, df_rf, df_rf_target]:
            d["tp6"] = np.log(d["tp6"].clip(lower=0)*1000 + 0.01)

        # ---------------------------------------------------
        # Add cyclical day-of-year columns to df_f, df_rf
        # (We typically only need them in the "features" data)
        # ---------------------------------------------------
        self._add_cyclical_day_of_year(df_f)
        self._add_cyclical_day_of_year(df_rf)

        # Restrict columns to the final feature list
        # (which now includes "cos_doy" and "sin_doy" if needed)
        df_f  = df_f[self.features]
        df_rf = df_rf[self.features]

        return df_f, df_f_target, df_rf, df_rf_target

    def validate_stations(self) -> bool:
        """
        Checks if self.stations_f and self.stations_rf have the same station_id ordering.
        """
        return (self.stations_f.station_id == self.stations_rf.station_id).all()

    @staticmethod
    def _add_cyclical_day_of_year(df: pd.DataFrame):
        """
        Utility to add cos_doy and sin_doy columns based on 'time'.
        """
        if "time" not in df.columns:
            return  # no time col => do nothing
        if not np.issubdtype(df["time"].dtype, np.datetime64):
            df["time"] = pd.to_datetime(df["time"])
        day_of_year = df["time"].dt.dayofyear
        df["cos_doy"] = np.cos(2*np.pi * day_of_year / 365.0)
        df["sin_doy"] = np.sin(2*np.pi * day_of_year / 365.0)


############################################################################
# 2. Distance & Graph-Building Helpers
############################################################################

def dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns geodesic distance (km) using geopy."""
    return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km

def compute_dist_matrix(stations: pd.DataFrame) -> np.ndarray:
    """
    Construct an NxN matrix of pairwise distances (km) from lat/lon columns.
    """
    coords = stations[["lat","lon"]].to_numpy()
    N = coords.shape[0]
    mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            mat[i,j] = dist_km(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
            mat[j,i] = mat[i,j]
    return mat

def build_edge_index_and_attr(dist_mat: np.ndarray, max_dist: float):
    """
    Edges where dist <= max_dist, plus self loops. Edge attr = 1/dist (normalized).
    """
    D = dist_mat.copy()
    np.fill_diagonal(D, np.inf)
    row, col = np.where(D <= max_dist)
    dist_vals = D[row, col]
    max_val   = dist_vals.max() if dist_vals.size > 0 else 1.0

    # Invert distance => heavier weight for closer stations
    inv_dist  = (dist_vals / max_val) ** -1

    edge_index = torch.tensor([row,col], dtype=torch.long)
    edge_attr  = torch.tensor(inv_dist, dtype=torch.float32).unsqueeze(-1)

    # Add self-loops
    N = dist_mat.shape[0]
    loops = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2,1)
    edge_index = torch.cat([edge_index, loops], dim=1)
    loop_attr  = torch.ones((N,1), dtype=torch.float32)
    edge_attr  = torch.cat([edge_attr, loop_attr], dim=0)

    return edge_index, edge_attr


def create_graphs_per_time(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    dist_mat: np.ndarray,
    max_dist: float
) -> List[Data]:
    """
    Creates one PyG Data object per unique time. 
      - data.x => first row of features per station [N, F_extra]
      - data.ensemble => entire ensemble => shape [N, E, F_pred]
      - data.y => [N] (tp6)
      - data.edge_index / data.edge_attr => from dist_mat
    """
    edge_index, edge_attr = build_edge_index_and_attr(dist_mat, max_dist)

    times = np.sort(df_features["time"].unique())
    graphs = []
    for t in times:
        # Subset for time t
        day_df    = df_features[df_features["time"] == t]
        day_target= df_targets[df_targets["time"] == t].sort_values("station_id")

        # Stations & ensemble members
        stations = day_df["station_id"].unique()
        N = len(stations)
        E = day_df["number"].nunique()

        exclude_cols = ["station_id","time","number"]
        feat_cols = [c for c in day_df.columns if c not in exclude_cols]

        # Extra features => first row per station
        first_feats = day_df.groupby("station_id").first()[feat_cols].to_numpy(dtype=np.float32)
        x_extra = torch.tensor(first_feats, dtype=torch.float32)

        # Ensemble => [N, E, #feat]
        sorted_df = day_df.sort_values(["station_id", "number"])
        arr = sorted_df[feat_cols].to_numpy(dtype=np.float32).reshape(N, E, -1)
        ens_t = torch.tensor(arr, dtype=torch.float32)

        # Targets
        y_vals = day_target["tp6"].to_numpy(dtype=np.float32)
        y_t  = torch.tensor(y_vals)

        data = Data(
            x=x_extra,
            ensemble=ens_t,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_t
        )
        data.timestamp = t
        graphs.append(data)

    return graphs


############################################################################
# 3. Main function: load, split, normalize, build graphs
############################################################################

def load_zarr_data_and_create_graphs(
    zarr_path: str,
    leadtime: str = "24h",
    max_dist: float = 100.0
) -> dict:
    """
    1) Load EUPPBench from Zarr,
    2) Build station distance matrix,
    3) Time-based split into train/test sets,
    4) Normalize (optional),
    5) Convert to PyG graphs (with cyclical day-of-year columns included).

    Returns:
       {
         "train":   List[Data],
         "test_rf": List[Data],
         "test_f":  List[Data],
         "stations": DataFrame  # optional
       }
    """
    # 1) Load from Zarr
    loader = ZarrLoader(data_path=zarr_path)
    df_f, df_f_target, df_rf, df_rf_target = loader.load_data(
        leadtime=leadtime,
        countries="all",
        features="all"  # includes cyclical columns: cos_doy, sin_doy
    )
    if not loader.validate_stations():
        raise ValueError("Forecast vs reforecast station mismatch!")

    # 2) Build station distance matrix from forecast stations
    stations_f = loader.stations_f
    dist_mat   = compute_dist_matrix(stations_f)

    # 3) Example time-based splitting
    train_cutoff = pd.Timestamp("2014-01-01")
    train_rf       = df_rf[df_rf["time"] < train_cutoff].copy()
    train_rf_target= df_rf_target[df_rf_target["time"] < train_cutoff].copy()
    test_rf        = df_rf[df_rf["time"] >= train_cutoff].copy()
    test_rf_target = df_rf_target[df_rf_target["time"] >= train_cutoff].copy()

    # For "test_f", we just use all forecast data
    test_f        = df_f.copy()
    test_f_target = df_f_target.copy()

    # 4) Normalize (fit on train, apply to test)
    sc = StandardScaler()
    exclude_cols = ["station_id","time","number"]
    numeric_cols = [c for c in train_rf.columns if c not in exclude_cols]

    train_rf.loc[:, numeric_cols] = sc.fit_transform(train_rf[numeric_cols])
    test_rf.loc[:, numeric_cols]  = sc.transform(test_rf[numeric_cols])
    test_f.loc[:, numeric_cols]   = sc.transform(test_f[numeric_cols])

    # 5) Create PyG graphs
    g_train   = create_graphs_per_time(train_rf,   train_rf_target,   dist_mat, max_dist)
    g_test_rf = create_graphs_per_time(test_rf,    test_rf_target,    dist_mat, max_dist)
    g_test_f  = create_graphs_per_time(test_f,     test_f_target,     dist_mat, max_dist)

    return {
        "train":    g_train,
        "test_rf":  g_test_rf,
        "test_f":   g_test_f,
        "stations": stations_f
    }


############################################################################
# 4. Additional Utility Functions
############################################################################

def split_graph(graph: Data, new_gnn=False) -> List[Data]:
    """
    Splits a graph which is created using 51 ensemble members into 5 subgraphs,
    each containing 10 or 11 ensemble members.
    """
    if new_gnn:
        # E.g. chunk [0-10, 10-20, 20-30, 30-40, 40-50]
        sets = [0, 10, 20, 30, 40, 50]
        subgraphs = []
        for i in range(5):
            g_copy = graph.clone()
            g_copy.ensemble = g_copy.ensemble[:, sets[i]:sets[i+1], :]
            subgraphs.append(g_copy)
        return subgraphs
    else:
        # Another approach: random permutation of 51 members
        # e.g. 51 * 122 nodes => rearr
        perm = torch.randperm(51) * 122
        idx  = perm[:, None] + torch.arange(122)
        set1 = idx[:10]; set2 = idx[10:20]; set3 = idx[20:30]
        set4 = idx[30:40]; set5 = idx[40:] # 11
        sets = [set1, set2, set3, set4, set5]
        out_graphs = []
        for s in sets:
            out_graphs.append(graph.subgraph(s.flatten()))
        return out_graphs


def shuffle_features(xs: torch.Tensor, feature_permute_idx: List[int]) -> torch.Tensor:
    """
    Shuffle a tensor of shape [T, N, F] in time & node dimension for selected feature indices.
    """
    xs_sub = xs[..., feature_permute_idx]
    T, N, _ = xs_sub.shape
    # Shuffle time dimension
    perm_t  = torch.randperm(T)
    xs_sub  = xs_sub[perm_t, ...]

    # Shuffle station dimension differently for each time
    idx = torch.argsort(torch.rand((T, N)), dim=1).unsqueeze(-1).expand(T, N, len(feature_permute_idx))
    out = torch.gather(xs_sub, dim=1, index=idx)
    xs[..., feature_permute_idx] = out
    return xs


def rm_edges(graphs: List[Data]) -> None:
    """
    Remove all edges (edge_index, edge_attr) from each PyG Data object.
    """
    for g in graphs:
        g.edge_index = torch.empty((2,0), dtype=torch.long)
        g.edge_attr  = torch.empty((0), dtype=torch.float32)


def summary_statistics(dataframes: defaultdict) -> defaultdict:
    """
    Example function: compute summary stats for each DataFrame except 'stations'.
    Replaces raw ensemble w/ (mean, std) per station. Then sets number=0.
    """
    only_mean = ["model_orography", "station_altitude", "station_latitude", "station_longitude"]
    for key, (df, tgt) in dataframes.items():
        if key == "stations":
            continue
        print(f"[INFO] Summaries for {key}")
        rest = [c for c in df.columns if c not in only_mean+["time","station_id","number"]]
        mean_agg = df.groupby(["time","station_id"])[only_mean].agg("mean")
        rest_agg = df.groupby(["time","station_id"])[rest].agg(["mean","std"])
        rest_agg.columns = ["_".join(c).strip() for c in rest_agg.columns.values]
        combined = pd.concat([mean_agg, rest_agg], axis=1).reset_index()
        combined["number"] = 0
        dataframes[key] = (combined, tgt)
    return dataframes
