from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
import xeofs as xe
import matplotlib.ticker as mticker

latex_font_size = 10

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": latex_font_size,
    "text.latex.preamble": r"\usepackage{lmodern}",
    "axes.labelsize": latex_font_size,
    "xtick.labelsize": latex_font_size,
    "ytick.labelsize": latex_font_size,
    "legend.fontsize": latex_font_size,
    "axes.titlesize": latex_font_size,
})


def kmeans_reanalysis(anomaly: xr.DataArray, n_clusters: int, tolerance: float, n_runs: int, fit_months: list | str = None) -> xr.Dataset:
    time_coords = anomaly["time"].values
    lat = anomaly["latitude"]
    lon = anomaly["longitude"]

    weights = np.sqrt(np.cos(np.deg2rad(anomaly['latitude'])))  # sqrt as frobenius norm is distance squared so the weighting of the distance is cos(\phi)
    weights /= weights.mean()  # rescale the weights so that the mean is 1 (probably has no effect but is standard practice)
    weights_2d = weights.broadcast_like(anomaly.isel(time=0))

    # print(weights_2d.shape)

    weights_flat = weights_2d.values.flatten()

    weighted_anomaly = anomaly * weights_2d
    weighted_anomaly = weighted_anomaly.transpose('time', 'latitude', 'longitude')
    nt, ny, nx = weighted_anomaly.shape

    data_flat = weighted_anomaly.values.reshape(nt, ny * nx)

    kmeans = sk.cluster.KMeans(
        n_clusters=n_clusters,
        n_init=n_runs,
        tol=tolerance,
        max_iter=500,
        random_state=0,
        verbose=0
    )

    # fitting logic for early or late winter of specific months
    if fit_months is not None:
        if fit_months == 'early_dec_jan_15':
            fit_mask = (anomaly["time"].dt.month == 12) | ((anomaly["time"].dt.day <= 15) & (anomaly["time"].dt.month == 1))
        elif fit_months == 'late_jan_16_feb':
            fit_mask = (anomaly["time"].dt.month == 2) | ((anomaly["time"].dt.day >= 16) & (anomaly["time"].dt.month == 1))
        elif isinstance(fit_months, (list, tuple, np.ndarray)):
            print(f'fitting months {fit_months}')
            fit_mask = anomaly["time"].dt.month.isin(fit_months)

        var_fit = weighted_anomaly.where(fit_mask, drop=True)
        data_fit = var_fit.values.reshape(-1, ny * nx)
        print(f'Check months being fit {np.unique(var_fit.time.dt.month.values)}')
        kmeans.fit(data_fit)

        assignments = kmeans.predict(data_flat)
    else:
        assignments = kmeans.fit_predict(data_flat)

    centers_weighted = kmeans.cluster_centers_  # These contain cos(lat)

    centers_physical = centers_weighted / weights_flat
    cluster_centers = centers_physical.reshape(n_clusters, ny, nx)

    center_norms = np.linalg.norm(centers_weighted, axis=1, keepdims=True)
    centers_weighted_norm = centers_weighted / center_norms

    amplitude_projection = data_flat @ centers_weighted_norm.T  # This is the Amplitude projection

    x_norm = data_flat / np.linalg.norm(data_flat, axis=1, keepdims=True)
    cosine_projection = x_norm @ centers_weighted_norm.T

    distances = kmeans.transform(data_flat).astype(np.float32)
    min_distances = np.min(distances, axis=1)

    return xr.Dataset(
        {
            "cluster_centres": xr.DataArray(
                cluster_centers,  # Now physical units (meters)
                dims=("cluster", "latitude", "longitude"),
                coords={"cluster": np.arange(n_clusters), "latitude": lat, "longitude": lon},
            ),
            "assignments": xr.DataArray(
                assignments,
                dims=("time",),
                coords={"time": time_coords},
            ),
            "amplitude_projections": xr.DataArray(
                amplitude_projection,
                dims=("time", "cluster"),
                coords={"time": time_coords, "cluster": np.arange(n_clusters)},
            ),
            "cosine_projections": xr.DataArray(
                cosine_projection,
                dims=("time", "cluster"),
                coords={"time": time_coords, "cluster": np.arange(n_clusters)},
            ),
            "distance_to_centres": xr.DataArray(
                distances,
                dims=("time", "cluster"),
                coords={"time": time_coords, "cluster": np.arange(n_clusters)},
            ),
            "min_distance": xr.DataArray(
                min_distances,
                dims=("time",),
                coords={"time": time_coords},
            ),
        }
    )


def assign_ensemble_forecast_to_reanalysis_clusters(forecast_anomaly: xr.DataArray, reanalysis_clusters: xr.Dataset) -> xr.Dataset:
    """
    Assigns SEAS5 members to the closest ERA5 cluster centroid.
    CRITICAL: Performs distance calculation in WEIGHTED space.
    """
    lat = forecast_anomaly['latitude']
    weights = np.sqrt(np.cos(np.deg2rad(lat)))
    weights /= weights.mean()
    weights_2d = weights.broadcast_like(forecast_anomaly.isel(time=0, number=0, step=0))  # (lat, lon)

    # (time, number, step, lat, lon) -> (N_samples, N_features)
    weighted_seas5 = forecast_anomaly * weights_2d
    weighted_seas5 = weighted_seas5.transpose('time', 'number', 'step', 'latitude', 'longitude')

    # dimensions
    nt, n_ens, n_step, ny, nx = weighted_seas5.shape
    n_features = ny * nx
    n_samples = nt * n_ens * n_step

    seas5_flat = weighted_seas5.values.reshape(n_samples, n_features)  # (N_samples, N_features)

    era5_centers_phys = reanalysis_clusters['cluster_centres'].values  # (n_clusters, lat, lon)
    weighted_centers = era5_centers_phys * weights_2d.values
    n_clusters = weighted_centers.shape[0]
    centers_flat = weighted_centers.reshape(n_clusters, n_features)

    center_norms = np.linalg.norm(centers_flat, axis=1, keepdims=True)
    centers_flat_norm = centers_flat / center_norms

    amplitude_projection = seas5_flat @ centers_flat_norm.T

    x_norm = seas5_flat / np.linalg.norm(seas5_flat, axis=1, keepdims=True)
    cosine_projection = x_norm @ centers_flat_norm.T

    dists = sp.spatial.distance.cdist(seas5_flat, centers_flat, metric='euclidean')

    assignments_flat = np.argmin(dists, axis=1)
    min_dists_flat = np.min(dists, axis=1)

    coords = {
        "time": forecast_anomaly.time,
        "number": forecast_anomaly.number,
        "step": forecast_anomaly.step,
        "valid_time": forecast_anomaly.valid_time
    }

    return xr.Dataset(
        {
            "assignments": xr.DataArray(
                assignments_flat.reshape(nt, n_ens, n_step),
                coords=coords,
                dims=("time", "number", "step")
            ),
            "amplitude_projections": xr.DataArray(
                amplitude_projection.reshape(nt, n_ens, n_step, n_clusters),
                coords={**coords, "cluster": np.arange(n_clusters)},
                dims=("time", "number", "step", "cluster")
            ),
            "cosine_projections": xr.DataArray(
                cosine_projection.reshape(nt, n_ens, n_step, n_clusters),
                coords={**coords, "cluster": np.arange(n_clusters)},
                dims=("time", "number", "step", "cluster")
            ),
            "distance_to_centres": xr.DataArray(
                dists.reshape(nt, n_ens, n_step, n_clusters),
                coords={**coords, "cluster": np.arange(n_clusters)},
                dims=("time", "number", "step", "cluster")
            ),
            # This is your "Similarity Index" source!
            "min_distance": xr.DataArray(
                min_dists_flat.reshape(nt, n_ens, n_step),
                coords=coords,
                dims=("time", "number", "step")
            )
        }
    )


def compute_reanalysis_spatial_mean_anomaly(ds: xr.Dataset, variable: str) -> xr.DataArray:
    # for z500 = ds[variable] / 9.80665 # m^2s^-2 -> m

    daily_clim = ds[variable].groupby("time.dayofyear").mean("time")  # daily climatology

    anomaly = ds[variable].groupby("time.dayofyear") - daily_clim  # daily anomaly

    anomaly = anomaly - anomaly.mean(dim=["latitude", "longitude"])  # remove spatial mean

    return anomaly.drop_vars("dayofyear")


def compute_reanalysis_linear_trend_anomaly(ds: xr.Dataset, variable: str) -> xr.DataArray:
    # for z500 = ds[variable] / 9.80665 # m^2s^-2 -> m

    daily_clim = ds[variable].groupby("time.dayofyear").mean("time")  # daily climatology

    anomaly = ds[variable].groupby("time.dayofyear") - daily_clim  # daily anomaly

    trend_coeffs = anomaly.polyfit(dim="time", deg=1)

    trend_line = xr.polyval(anomaly.time, trend_coeffs.polyfit_coefficients)  # trend line

    detrended_anomaly = anomaly - trend_line

    return detrended_anomaly.drop_vars("dayofyear")


def compute_reanalysis_zscore_variable(ds: xr.Dataset, variable: str) -> xr.DataArray:
    # for z500 = ds[variable] / 9.80665 # m^2s^-2 -> m

    climatology = ds[variable].mean("time")
    std = ds[variable].std("time")
    zscore = (ds[variable] - climatology) / std

    return zscore


def compute_reanalysis_fixed_anomaly(ds: xr.Dataset, variable: str) -> xr.DataArray:
    # for z500 = ds[variable] / 9.80665 # m^2s^-2 -> m

    climatology = ds[variable].mean("time")  # fixed climatology
    anomaly = ds[variable] - climatology

    return anomaly


def compute_ensemble_zscore_variable(ds: xr.Dataset, variable: str) -> xr.DataArray:
    # z500 = ds[variable] / 9.80665  # m^2s^-2 -> m

    climatology = ds[variable].mean(dim=["time", "number", "step"]).astype(np.float32)
    std = ds[variable].std(dim=["time", "number", "step"]).astype(np.float32)
    zscore = ((ds[variable] - climatology) / std).astype(np.float32)
    return zscore


def compute_ensemble_spatial_mean_anomaly(ds: xr.Dataset, variable: str) -> xr.DataArray:
    # z500 = ds[variable] / 9.80665  # m^2s^-2 -> m

    step_climatology = ds[variable].mean(dim=["time", "number"]).astype(np.float32)  # daily clim
    anomaly = (ds[variable] - step_climatology).astype(np.float32)
    anomaly = (anomaly - anomaly.mean(dim=["latitude", "longitude"])).astype(np.float32)

    return anomaly


def compute_ensemble_linear_trend_anomaly(ds: xr.Dataset, variable: str) -> xr.DataArray:
    # z500 = ds[variable] / 9.80665  # m^2s^-2 -> m

    step_climatology = ds[variable].mean(dim=["time", "number"]).astype(np.float32)  # daily clim
    anomaly = (ds[variable] - step_climatology).astype(np.float32)
    ens_mean_anom = anomaly.mean(dim="number")

    trend_coeffs = ens_mean_anom.polyfit(dim="time", deg=1)
    trend_line = xr.polyval(ds.time, trend_coeffs.polyfit_coefficients)

    # xarray does the broadcasting automatically: (Time, Number, Step, Lat, Lon) - (Time, Step, Lat, Lon)
    detrended_anomaly = anomaly - trend_line

    return (detrended_anomaly).astype(np.float32)


def compute_ensemble_fixed_anomaly(ds: xr.Dataset, variable: str) -> xr.DataArray:
    # z500 = ds[variable] / 9.80665  # m^2s^-2 -> m

    climatology = ds[variable].mean(dim=["time", "number", "step"]).astype(np.float32)
    anomaly = (ds[variable] - climatology).astype(np.float32)
    return anomaly


def compute_era5_modes(era5_anom, n_modes=2):

    drop_coords = [
        c for c in era5_anom.coords
        if c not in ("latitude", "longitude", "time")
    ]
    era5_clean = era5_anom.drop_vars(drop_coords)
    model = xe.single.EOF(n_modes=n_modes, use_coslat=True)
    model.fit(era5_clean, dim="time")

    return model


def project_seas5_modes(seas5_anom, era5_model, era5_anom):

    s5_safe = seas5_anom.rename({"time": "init_time"})  # time is different in seas5 and era5 so have to rename to init_time which is what it actually is

    s5_stacked = s5_safe.stack(sample=("init_time", "number", "step"))

    s5_simple = s5_stacked.reset_index("sample")

    s5_ready = s5_simple.rename({"sample": "time"})

    drop_coords = [
        c for c in era5_anom.coords
        if c not in ("latitude", "longitude", "time")
    ]
    era5_clean = era5_anom.drop_vars(drop_coords)

    template = (
        era5_clean
        .isel(time=0, drop=True)
        .drop_vars([
            c for c in era5_clean.coords
            if c not in ("latitude", "longitude", "time")
        ])
    )

    s5_aligned = s5_ready.reindex_like(template, method=None)
    s5_aligned = s5_aligned.transpose("time", "latitude", "longitude")
    s5_clean = s5_aligned.drop_vars(
        [c for c in s5_aligned.coords if c not in ("latitude", "longitude", "time")]
    )
    pcs_projected = era5_model.transform(s5_clean)

    pcs_flat = pcs_projected.rename({"time": "sample"})
    pcs_flat = pcs_flat.assign_coords(sample=s5_stacked.sample)
    pcs_out = pcs_flat.unstack("sample")

    return pcs_out.rename({"init_time": "time"})


def compute_regression_stats(seas5_da, era5_da):
    """
    seas5_da: (time, number)
    era5_da:  (time)
    Returns: slope, r, r2, p
    """
    from scipy.stats import linregress

    # Ensemble mean first
    seas5_mean = seas5_da.mean(dim="number")

    y = era5_da.values
    x = seas5_mean.values

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan

    result = linregress(x[mask], y[mask])

    return result.slope, result.rvalue, result.rvalue**2, result.pvalue


def compute_fixed_winter_month_metrics(
    era5_clusters: xr.Dataset,
    seas5_clusters: xr.Dataset,
    era5_pcs: xr.DataArray,
    seas5_pcs: xr.DataArray,
    winter_definitions: dict,
    era5_cluster_order: list,
    seas5_cluster_order: list,
    cluster_names: list,
) -> xr.Dataset:

    all_means = []
    regime_stats_list = []
    eof_stats_list = []

    valid_time = seas5_clusters.valid_time
    era5_aligned = era5_clusters.sel(time=valid_time)
    era5_pcs_aligned = era5_pcs.sel(time=valid_time)

    seas5_dist = seas5_clusters["distance_to_centres"]
    era5_dist = era5_aligned["distance_to_centres"]

    seas5_dist_std = - (seas5_dist - seas5_dist.mean(dim="cluster")) / seas5_dist.std(dim="cluster")
    era5_dist_std = - (era5_dist - era5_dist.mean(dim="cluster")) / era5_dist.std(dim="cluster")

    for winter_name, months in winter_definitions.items():

        print(f"Computing {winter_name}")

        month_mask = valid_time.dt.month.isin(months)

        seas5_freq_list = []
        era5_freq_list = []

        for k in era5_clusters.cluster.values:

            seas5_hits = (seas5_clusters["assignments"] == seas5_cluster_order[k])
            era5_hits = (era5_aligned["assignments"] == era5_cluster_order[k])

            seas5_freq = seas5_hits.where(month_mask).mean(dim="step")
            era5_freq = era5_hits.where(month_mask).mean(dim="step")

            seas5_freq_list.append(seas5_freq)
            era5_freq_list.append(era5_freq)

        seas5_freq = xr.concat(seas5_freq_list, dim="cluster").assign_coords(cluster=cluster_names)
        era5_freq = xr.concat(era5_freq_list, dim="cluster").assign_coords(cluster=cluster_names)

        seas5_proj_unordered = seas5_clusters["amplitude_projections"].where(month_mask).mean(dim="step")
        era5_proj_unordered = era5_aligned["amplitude_projections"].where(month_mask).mean(dim="step")
        seas5_proj = seas5_proj_unordered.sel(cluster=seas5_cluster_order).assign_coords(cluster=cluster_names)
        era5_proj = era5_proj_unordered.sel(cluster=era5_cluster_order).assign_coords(cluster=cluster_names)

        seas5_index_unordered = seas5_dist_std.where(month_mask).mean(dim="step")
        era5_index_unordered = era5_dist_std.where(month_mask).mean(dim="step")
        seas5_index = seas5_index_unordered.sel(cluster=seas5_cluster_order).assign_coords(cluster=cluster_names)
        era5_index = era5_index_unordered.sel(cluster=era5_cluster_order).assign_coords(cluster=cluster_names)

        seas5_nao = seas5_pcs.sel(mode=1).where(month_mask).mean(dim="step").drop_vars('mode')
        era5_nao = era5_pcs_aligned.sel(mode=1).where(month_mask).mean(dim="step").drop_vars('mode')

        seas5_eap = seas5_pcs.sel(mode=2).where(month_mask).mean(dim="step").drop_vars('mode')
        era5_eap = era5_pcs_aligned.sel(mode=2).where(month_mask).mean(dim="step").drop_vars('mode')

        # print(type(seas5_freq),type(era5_freq),type(seas5_proj),type(era5_proj),type(seas5_index),type(era5_index),type(seas5_nao),type(era5_nao),type(seas5_eap),type(era5_eap))
        means_ds = xr.Dataset({
            "seas5_frequency": seas5_freq,
            "era5_frequency": era5_freq,
            "seas5_projection": seas5_proj,
            "era5_projection": era5_proj,
            "seas5_index": seas5_index,
            "era5_index": era5_index,
            "seas5_nao": seas5_nao,
            "era5_nao": era5_nao,
            "seas5_eap": seas5_eap,
            "era5_eap": era5_eap,
        }).expand_dims(winter=[winter_name])

        all_means.append(means_ds)

        regime_metrics = {
            "frequency": (seas5_freq, era5_freq),
            "projection": (seas5_proj, era5_proj),
            "index": (seas5_index, era5_index),
        }

        current_winter_regime_stats = []

        for metric_name, (s5, e5) in regime_metrics.items():

            stat_arr = []

            for k_name in cluster_names:

                slope, r, r2, p = compute_regression_stats(
                    s5.sel(cluster=k_name),
                    e5.sel(cluster=k_name)
                )

                stat_arr.append([slope, r, r2, p])

            stat_arr = np.array(stat_arr)

            stats_ds = xr.Dataset(
                {
                    "regime_stats": (
                        ["cluster", "stat"],
                        stat_arr
                    )
                },
                coords={
                    "cluster": cluster_names,
                    "stat": ["slope", "r", "r2", "p"],
                    "metric": metric_name,
                    "winter": winter_name,
                }
            )
            current_winter_regime_stats.append(stats_ds.expand_dims("metric"))
        ds_regime_winter = xr.concat(current_winter_regime_stats, dim="metric")
        regime_stats_list.append(ds_regime_winter.expand_dims(winter=[winter_name]))

        eof_modes = {
            "nao": (seas5_nao, era5_nao),
            "eap": (seas5_eap, era5_eap),
        }

        current_winter_eof_stats = []

        for mode, (s5, e5) in eof_modes.items():

            slope, r, r2, p = compute_regression_stats(s5, e5)

            stat_arr = np.array([slope, r, r2, p])

            stats_ds = xr.Dataset(
                {
                    "eof_stats": (
                        ["stat"],
                        stat_arr
                    )
                },
                coords={
                    "stat": ["slope", "r", "r2", "p"],
                    "mode": mode,
                    "winter": winter_name,
                }
            )

            current_winter_eof_stats.append(stats_ds.expand_dims("mode"))
        ds_eof_winter = xr.concat(current_winter_eof_stats, dim="mode")
        eof_stats_list.append(ds_eof_winter.expand_dims(winter=[winter_name]))

    means_final = xr.concat(all_means, dim="winter")
    regime_stats_final = xr.concat(regime_stats_list, dim="winter")
    eof_stats_final = xr.concat(eof_stats_list, dim="winter")

    return xr.merge([means_final, regime_stats_final, eof_stats_final])
