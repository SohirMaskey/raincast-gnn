import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
from typing import Tuple, List, Union
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
