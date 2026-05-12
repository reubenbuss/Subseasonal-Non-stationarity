import numpy as np
import xarray as xr
from tqdm import tqdm


def batch_mediation(y, x, m):

    # mean centring
    y_anom = y - np.nanmean(y, axis=1, keepdims=True)
    x_anom = x - np.nanmean(x, axis=1, keepdims=True)
    m_anom = m - np.nanmean(m, axis=1, keepdims=True)

    # to standardise for each window for SPV
    # y_anom = (y - np.nanmean(y, axis=1, keepdims=True)) / (np.nanstd(y, axis=1, keepdims=True))
    # x_anom = (x - np.nanmean(x, axis=1, keepdims=True)) / (np.nanstd(x, axis=1, keepdims=True))
    # m_anom = (m - np.nanmean(m, axis=1, keepdims=True)) / (np.nanstd(m, axis=1, keepdims=True))

    # sum of squares
    ss_x = np.nansum(x_anom**2, axis=1)
    ss_m = np.nansum(m_anom**2, axis=1)

    # sum of products
    sp_xy = np.nansum(x_anom * y_anom, axis=1)
    sp_xm = np.nansum(x_anom * m_anom, axis=1)
    sp_my = np.nansum(m_anom * y_anom, axis=1)

    # Pathway: Total Effect (Regime = b0 + b1*ENSO + e)
    total = sp_xy / ss_x

    # Pathway: ENSO->SPV (SPV = c0 + c1*ENSO + e')
    es_path = sp_xm / ss_x

    # Pathway: Direct & SPV -> Regime (Regime = a0 + a1*ENSO + a2*SPV + e'')
    denom = (ss_x * ss_m) - (sp_xm**2)
    denom = np.where(denom == 0, np.nan, denom)  # Prevent division by zero

    direct = (sp_xy * ss_m - sp_my * sp_xm) / denom
    sr_path = (sp_my * ss_x - sp_xy * sp_xm) / denom

    # Pathway: Indirect Effect
    indirect = es_path * sr_path

    return np.column_stack([total, direct, indirect, es_path, sr_path])


def ensemble_batch_mediation(y, x, m, sr_path):
    #

    # mean centring
    y_anom = y - np.nanmean(y, axis=1, keepdims=True)
    x_anom = x - np.nanmean(x, axis=1, keepdims=True)
    m_anom = m - np.nanmean(m, axis=1, keepdims=True)

    # y_anom = (y - np.nanmean(y, axis=1, keepdims=True)) / (np.nanstd(y, axis=1, keepdims=True))
    # x_anom = (x - np.nanmean(x, axis=1, keepdims=True)) / (np.nanstd(x, axis=1, keepdims=True))
    # m_anom = (m - np.nanmean(m, axis=1, keepdims=True)) / (np.nanstd(m, axis=1, keepdims=True))

    # sum of squares
    ss_x = np.nansum(x_anom**2, axis=1)
    ss_m = np.nansum(m_anom**2, axis=1)

    # sum of products
    sp_xy = np.nansum(x_anom * y_anom, axis=1)
    sp_xm = np.nansum(x_anom * m_anom, axis=1)
    sp_my = np.nansum(m_anom * y_anom, axis=1)

    # Pathway: Total Effect (Regime = b0 + b1*ENSO + e)
    total = sp_xy / ss_x

    # Pathway: ENSO->SPV (SPV = c0 + c1*ENSO + e')
    es_path = sp_xm / ss_x

    # Pathway: Indirect Effect
    indirect = es_path * sr_path

    return np.column_stack([total, indirect, es_path])


def run_temporal_bootstrap_mediation(ds, y_var, x_var, m_var, n_boot=2000, ensemble=True, seed=42):
    rng = np.random.default_rng(seed)
    windows = ds.window.values
    clusters = ds.cluster.values

    coeffs = ["Total", "Direct", "Indirect", r"ENSO → SPV", r"SPV → Regime"]

    mean_arr = np.full((len(clusters), len(windows), 5), np.nan)
    sig_arr = np.full((len(clusters), len(windows), 5), False)

    for j, cluster in enumerate(clusters):
        for i, window in enumerate(windows):
            ds_w = ds.sel(window=window)

            if ensemble:
                y = ds_w[y_var].sel(cluster=cluster).mean("number").values
                x = ds_w[x_var].mean("number").values
                m = ds_w[m_var].mean("number").values
            else:
                y = ds_w[y_var].sel(cluster=cluster).values
                x = ds_w[x_var].values
                m = ds_w[m_var].values

            n_years = len(y)

            idx = rng.choice(n_years, size=(n_boot, n_years), replace=True)

            boot_coeffs = batch_mediation(y[idx], x[idx], m[idx])

            mean_arr[j, i, :] = np.nanmean(boot_coeffs, axis=0)
            lower = np.nanpercentile(boot_coeffs, 2.5, axis=0)
            upper = np.nanpercentile(boot_coeffs, 97.5, axis=0)
            sig_arr[j, i, :] = (lower > 0) | (upper < 0)

    return xr.Dataset(
        {
            "mediation_mean": (("cluster", "window", "pathway"), mean_arr),
            "mediation_sig": (("cluster", "window", "pathway"), sig_arr),
        },
        coords={"cluster": clusters, "window": windows, "pathway": coeffs}
    )


def run_member_sampling_mediation(ds, y_var, x_var, m_var, n_iterations=2000, seed=42):

    rng = np.random.default_rng(seed)
    windows = ds.window.values
    clusters = ds.cluster.values
    n_years = ds.sizes['time']
    n_members = ds.sizes['number']

    coeffs = ["Total", "Direct", "Indirect", r"ENSO → SPV", r"SPV → Regime"]

    all_coeffs_arr = np.full((n_iterations, len(clusters), len(windows), 5), np.nan)

    # random member index shape (n_iterations, n_years)
    member_idx = rng.integers(0, n_members, size=(n_iterations, n_years))
    # random winter index shape (n_iterations, n_years)
    time_idx = rng.integers(0, n_years, size=(n_iterations, n_years))

    for j, cluster in enumerate(tqdm(clusters, desc="Clusters")):
        for i, window in enumerate(windows):

            ds_w = ds.sel(window=window)

            y_full = ds_w[y_var].sel(cluster=cluster).values
            x_full = ds_w[x_var].values
            m_full = ds_w[m_var].values

            # print(np.shape(y_full)) # (44, 51)

            y_synth = y_full[time_idx, member_idx]
            x_synth = x_full[time_idx, member_idx]
            m_synth = m_full[time_idx, member_idx]
            # print(np.shape(y_synth)) # (2000, 44)

            all_coeffs_arr[:, j, i, :] = batch_mediation(y_synth, x_synth, m_synth)

    return xr.Dataset(
        {
            "mediation_coeffs": (("iteration", "cluster", "window", "pathway"), all_coeffs_arr),
        },
        coords={
            "iteration": np.arange(n_iterations),
            "cluster": clusters,
            "window": windows,
            "pathway": coeffs
        }
    )


def run_pooled_bootstrap_mediation_mean(ds, y_var, x_var, m_var, n_boot=2000, seed=42):

    rng = np.random.default_rng(seed)
    windows = ds.window.values
    clusters = ds.cluster.values

    n_years = ds.sizes['time']
    n_members = ds.sizes['number'] if 'number' in ds.dims else 1

    coeffs = ["Total", "Direct", "Indirect", r"ENSO → SPV", r"SPV → Regime"]

    mean_arr = np.full((len(clusters), len(windows), 5), np.nan)
    sig_arr = np.full((len(clusters), len(windows), 5), False)

    year_idx = rng.choice(n_years, size=(n_boot, n_years), replace=True)

    for j, cluster in enumerate(tqdm(clusters, desc="Clusters")):
        for i, window in enumerate(windows):
            ds_w = ds.sel(window=window)

            y_full = ds_w[y_var].sel(cluster=cluster).values
            x_full = ds_w[x_var].values
            m_full = ds_w[m_var].values

            if y_full.ndim > 1:
                # print(y_full.shape, 'seas5')

                y_boot = y_full[year_idx, :]  # regime
                x_boot = x_full[year_idx, :]  # enso
                m_boot = m_full[year_idx, :]  # spv

                # y_boot (2000,44,51) - > (2000, 44*51)
                # print(y_boot.shape)
                y_flat = y_boot.reshape(n_boot, -1)
                x_flat = x_boot.reshape(n_boot, -1)
                m_flat = m_boot.reshape(n_boot, -1)

                ym_boot = batch_mediation(y_flat, x_flat, m_flat)

                ym_direct = ym_boot[:, 1]
                ym_sr_path = ym_boot[:, 4]

                # print(y_boot.shape)

                y_ens_mean = np.nanmean(y_boot, axis=2)
                x_ens_mean = np.nanmean(x_boot, axis=2)
                m_ens_mean = np.nanmean(m_boot, axis=2)

                row_valid = np.sum(np.isfinite(y_ens_mean), axis=1)

                y_boots = ensemble_batch_mediation(y_ens_mean, x_ens_mean, m_ens_mean, ym_sr_path)

                y_total = y_boots[:, 0]
                y_m_indirect = y_boots[:, 1]
                y_es_path = y_boots[:, 2]

                boot_coeffs = np.column_stack([
                    y_total,        # temporal total
                    ym_direct,      # temporal & member direct
                    y_m_indirect,     # temporal beta * temporal & member gamma indirect
                    y_es_path,      # temporal ENSO->SPV
                    ym_sr_path      # temporal & member bootstraps SPV->regime
                ])

            else:
                # print(y_full.ndim, 'era5')
                y_boot = y_full[year_idx]  # regime
                x_boot = x_full[year_idx]  # enso
                m_boot = m_full[year_idx]  # spv
                boot_coeffs = batch_mediation(y_boot, x_boot, m_boot)

            mean_arr[j, i, :] = np.nanmean(boot_coeffs, axis=0)
            lower = np.nanpercentile(boot_coeffs, 2.5, axis=0)
            upper = np.nanpercentile(boot_coeffs, 97.5, axis=0)
            sig_arr[j, i, :] = (lower > 0) | (upper < 0)

    return xr.Dataset(
        {
            "mediation_mean": (("cluster", "window", "pathway"), mean_arr),
            "mediation_sig": (("cluster", "window", "pathway"), sig_arr),
        },
        coords={"cluster": clusters, "window": windows, "pathway": coeffs}
    )


def run_pooled_bootstrap_mediation(ds, y_var, x_var, m_var, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    windows = ds.window.values
    clusters = ds.cluster.values

    n_years = ds.sizes['time']
    n_members = ds.sizes['number'] if 'number' in ds.dims else 1

    coeffs = ["Total", "Direct", "Indirect", r"ENSO → SPV", r"SPV → Regime"]

    # Store ALL iterations! Shape: (2000, n_clusters, n_windows, 5)
    boot_arr = np.full((n_boot, len(clusters), len(windows), 5), np.nan)

    year_idx = rng.choice(n_years, size=(n_boot, n_years), replace=True)

    for j, cluster in enumerate(tqdm(clusters, desc="Clusters")):
        for i, window in enumerate(windows):
            ds_w = ds.sel(window=window)

            y_full = ds_w[y_var].sel(cluster=cluster).values
            x_full = ds_w[x_var].values
            m_full = ds_w[m_var].values

            if y_full.ndim == 1:
                y_full = np.tile(y_full[:, None], (1, n_members))
            if x_full.ndim == 1:
                x_full = np.tile(x_full[:, None], (1, n_members))
            if m_full.ndim == 1:
                m_full = np.tile(m_full[:, None], (1, n_members))

            y_boot = y_full[year_idx, :]
            x_boot = x_full[year_idx, :]
            m_boot = m_full[year_idx, :]

            y_flat = y_boot.reshape(n_boot, -1)
            x_flat = x_boot.reshape(n_boot, -1)
            m_flat = m_boot.reshape(n_boot, -1)

            # Store the full 2000 iterations directly into the array
            boot_arr[:, j, i, :] = batch_mediation(y_flat, x_flat, m_flat)

    return xr.Dataset(
        {
            "mediation_coeffs": (("iteration", "cluster", "window", "pathway"), boot_arr),
        },
        coords={
            "iteration": np.arange(n_boot),
            "cluster": clusters,
            "window": windows,
            "pathway": coeffs
        }
    )


def main():
    print('Ooops')


if __name__ == "__main__":
    main()
