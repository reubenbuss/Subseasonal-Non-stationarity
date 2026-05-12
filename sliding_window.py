from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib as mpl
import time
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpecFromSubplotSpec
from IPython.display import display, Markdown
import calendar
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize


def compute_sliding_window_means(
    era5_clusters: xr.Dataset,
    seas5_clusters: xr.Dataset,
    era5_pcs: xr.DataArray,
    seas5_pcs: xr.DataArray,
    era5_u10: xr.DataArray,
    seas5_u10: xr.DataArray,
    era5_spv: xr.DataArray,
    seas5_spv: xr.DataArray,
    era5_enso: xr.DataArray,
    seas5_enso: xr.DataArray,
    era5_cluster_order: list,
    seas5_cluster_order: list,
    cluster_names: list,
    window_size: int = 45,
    extended=False
) -> xr.Dataset:

    lagged_valid_time = seas5_spv.valid_time
    DJF_valid_time = seas5_clusters.valid_time

    seas5_u10 = seas5_u10.where(lagged_valid_time, drop=True)
    seas5_spv = seas5_spv.where(lagged_valid_time, drop=True)
    seas5_enso = seas5_enso.where(DJF_valid_time, drop=True)
    seas5_clusters = seas5_clusters.where(DJF_valid_time, drop=True)
    seas5_pcs = seas5_pcs.where(DJF_valid_time, drop=True)

    seas5_spv['step'] = seas5_clusters.step
    seas5_u10['step'] = seas5_clusters.step
    seas5_spv['valid_time'] = DJF_valid_time
    seas5_u10['valid_time'] = DJF_valid_time

    if extended:
        init_month = int(seas5_spv.time.dt.month[0])
        init_day = int(seas5_spv.time.dt.day[0])
        start_year = int(era5_spv.time.dt.year.min())
        end_year = int(era5_spv.time.dt.year.max())

        pseudo_inits = pd.to_datetime([f"{y}-{init_month:02d}-{init_day:02d}" for y in range(start_year, end_year)])
        era5_time_dim = xr.DataArray(pseudo_inits, dims=['time'], name='time')

        spv_timedeltas = seas5_spv.valid_time.isel(time=0) - seas5_spv.time.isel(time=0)
        cluster_timedeltas = seas5_clusters.valid_time.isel(time=0) - seas5_clusters.time.isel(time=0)

        extended_lagged_valid_time = era5_time_dim + spv_timedeltas
        extended_DJF_valid_time = era5_time_dim + cluster_timedeltas

        era5_aligned = era5_clusters.sel(time=extended_DJF_valid_time)
        era5_pcs_aligned = era5_pcs.sel(time=extended_DJF_valid_time)
        era5_u10_aligned = era5_u10.sel(time=extended_lagged_valid_time)
        era5_spv_aligned = era5_spv.sel(time=extended_lagged_valid_time)
        era5_enso_aligned = era5_enso.sel(time=extended_DJF_valid_time)
        era5_u10_aligned['step'] = era5_aligned.step
        era5_spv_aligned['step'] = era5_aligned.step
        era5_u10_aligned['valid_time'] = extended_DJF_valid_time
        era5_spv_aligned['valid_time'] = extended_DJF_valid_time

    else:
        era5_aligned = era5_clusters.sel(time=DJF_valid_time)
        era5_pcs_aligned = era5_pcs.sel(time=DJF_valid_time)
        era5_u10_aligned = era5_u10.sel(time=lagged_valid_time)
        era5_spv_aligned = era5_spv.sel(time=lagged_valid_time)
        era5_enso_aligned = era5_enso.sel(time=DJF_valid_time)
        era5_u10_aligned['step'] = era5_aligned.step
        era5_spv_aligned['step'] = era5_aligned.step
        era5_u10_aligned['valid_time'] = DJF_valid_time
        era5_spv_aligned['valid_time'] = DJF_valid_time

    seas5_dist = seas5_clusters["distance_to_centres"]
    era5_dist = era5_aligned["distance_to_centres"]

    seas5_index = - (seas5_dist - seas5_dist.mean("cluster")) / seas5_dist.std("cluster")
    era5_index = - (era5_dist - era5_dist.mean("cluster")) / era5_dist.std("cluster")

    roll_kwargs = dict(step=window_size, center=True)

    seas5_proj_roll = seas5_clusters["projections"].rolling(**roll_kwargs).mean().dropna("step")
    era5_proj_roll = era5_aligned["Projections"].rolling(**roll_kwargs).mean().dropna("step")

    seas5_index_roll = seas5_index.rolling(**roll_kwargs).mean().dropna("step")
    era5_index_roll = era5_index.rolling(**roll_kwargs).mean().dropna("step")

    seas5_nao_roll = seas5_pcs.sel(mode=1).drop_vars("mode").rolling(**roll_kwargs).mean().dropna("step")
    era5_nao_roll = era5_pcs_aligned.sel(mode=1).drop_vars("mode").rolling(**roll_kwargs).mean().dropna("step")

    seas5_eap_roll = seas5_pcs.rolling(**roll_kwargs).mean().dropna("step")
    era5_eap_roll = era5_pcs_aligned.rolling(**roll_kwargs).mean().dropna("step")

    seas5_u10_roll = seas5_u10.rolling(**roll_kwargs).mean().dropna("step")
    era5_u10_roll = era5_u10_aligned.rolling(**roll_kwargs).mean().dropna("step")

    seas5_spv_roll = seas5_spv.rolling(**roll_kwargs).mean().dropna("step")
    era5_spv_roll = era5_spv_aligned.rolling(**roll_kwargs).mean().dropna("step")

    seas5_enso_roll = seas5_enso.rolling(**roll_kwargs).mean().dropna("step")
    era5_enso_roll = era5_enso_aligned.rolling(**roll_kwargs).mean().dropna("step")

    # print(seas5_proj_roll.valid_time)
    # print(era5_proj_roll.valid_time)

    seas5_freq_list = []
    era5_freq_list = []

    for k in range(len(cluster_names)):

        seas5_hits = (seas5_clusters["assignments"] == seas5_cluster_order[k])
        era5_hits = (era5_aligned["assignments"] == era5_cluster_order[k])

        seas5_freq_roll = seas5_hits.rolling(**roll_kwargs).mean().dropna("step")
        era5_freq_roll = era5_hits.rolling(**roll_kwargs).mean().dropna("step")

        seas5_freq_list.append(seas5_freq_roll)
        era5_freq_list.append(era5_freq_roll)

    seas5_freq_roll = xr.concat(seas5_freq_list, dim="cluster").assign_coords(cluster=cluster_names)
    era5_freq_roll = xr.concat(era5_freq_list, dim="cluster").assign_coords(cluster=cluster_names)

    # order clusters
    seas5_proj_roll = seas5_proj_roll.sel(cluster=seas5_cluster_order).assign_coords(cluster=cluster_names)
    era5_proj_roll = era5_proj_roll.sel(cluster=era5_cluster_order).assign_coords(cluster=cluster_names)

    seas5_index_roll = seas5_index_roll.sel(cluster=seas5_cluster_order).assign_coords(cluster=cluster_names)
    era5_index_roll = era5_index_roll.sel(cluster=era5_cluster_order).assign_coords(cluster=cluster_names)

    def drop_valid_time(da):
        return da.reset_coords("valid_time", drop=True)

    means_ds = xr.Dataset({
        "seas5_frequency": drop_valid_time(seas5_freq_roll),
        "era5_frequency": drop_valid_time(era5_freq_roll),
        "seas5_projection": drop_valid_time(seas5_proj_roll),
        "era5_projection": drop_valid_time(era5_proj_roll),
        "seas5_index": drop_valid_time(seas5_index_roll),
        "era5_index": drop_valid_time(era5_index_roll),
        "seas5_nao": drop_valid_time(seas5_nao_roll),
        "era5_nao": drop_valid_time(era5_nao_roll),
        "seas5_eap": drop_valid_time(seas5_eap_roll),
        "era5_eap": drop_valid_time(era5_eap_roll),
        "seas5_u10": drop_valid_time(seas5_u10_roll),
        "era5_u10": drop_valid_time(era5_u10_roll),
        "seas5_spv": drop_valid_time(seas5_spv_roll),
        "era5_spv": drop_valid_time(era5_spv_roll),
        "seas5_enso": drop_valid_time(seas5_enso_roll),
        "era5_enso": drop_valid_time(era5_enso_roll),
    })

    half = window_size // 2

    window_time = DJF_valid_time.isel(
        time=0,
        step=slice(half, -half)
    )

    window_label = window_time.dt.strftime("%m-%d")

    means_ds = means_ds.assign_coords(
        window=("step", window_label.values)
    ).swap_dims({"step": "window"})

    return means_ds


def plot_skill_window_heatmaps(
    data,
    sig,
    cluster_names=None,
    skill="corr",
    file_path=None,
    dataset_name="TEST",
    window_size='days'
):
    """
    Plot mediation heatmaps for sliding-window mediation results.

    dataset must contain:
        mediation_mean(window, cluster, pathway)
        mediation_sig(window, cluster, pathway)
    """

    windows = data.window.values
    clusters = data.cluster.values

    # shape -> (cluster, window)
    data = data.transpose("cluster", "window").values
    sig = sig.transpose("cluster", "window").values

    if skill == 'corr':
        vmax = 0.5
        vmin = -vmax
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        ticks = np.linspace(vmin, vmax, 3)
        cmap = 'RdBu_r'
    else:
        vmax = 3
        vmin = -1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
        ticks = np.linspace(vmin, vmax, 5)
        cmap = 'PiYG'

    fig, ax = plt.subplots(figsize=(3.5, 1.4))
    # fig, ax = plt.subplots(figsize=(3.5, 1.5))

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation='none'
    )

    # significance dots
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if sig[i, j]:
                ax.plot(j, i, marker="_", c='k', markersize=2)

    # Y axis
    if cluster_names is None:
        cluster_names = clusters

    ax.set_yticks(np.arange(len(cluster_names)))
    ax.set_yticklabels(cluster_names, fontsize=10)

    fig.colorbar(im, ax=ax, pad=0.02, location='right', orientation='vertical', ticks=ticks)

    # # X axis
    # spacing = max(len(windows) // 6, 1)
    # ax.set_xticks(np.arange(0, len(windows), spacing))
    # ax.set_xticklabels(windows[::spacing], rotation=45)

    # ['12-01','01-01','']
    # ax.set_xticks(np.arange(0, len(windows), spacing))
    # ax.set_xticklabels(windows[::spacing], rotation=45)

    # ax.tick_params(axis='both', which='major', labelsize=10)

    major_labels = ['01-01', '02-01', '03-01']
    minor_labels = ['12-15', '01-15', '02-15', '03-15']
    minor_positions = np.where(np.isin(windows, minor_labels))[0]
    major_positions = np.where(np.isin(windows, major_labels))[0]

    ax.set_xticks(major_positions)
    ax.set_xticklabels(major_labels, rotation=0, ha='right', fontsize=10)

    ax.tick_params(axis='y', which='major', labelsize=10)

    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_positions))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(minor_labels))
    ax.tick_params(axis='x', which='minor', length=5, labelsize=8, rotation=0)
    ax.tick_params(axis='x', which='major', length=12, pad=6)

    for label in ax.get_xminorticklabels():
        label.set_ha('center')

    for label in ax.get_xmajorticklabels():
        label.set_ha('center')

    ax.set_title(f"{dataset_name}", fontsize=10)

    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.26, top=0.85)
    # fig.subplots_adjust(left=0.15, right=0.99, bottom=0.26, top=0.76)

    if file_path is not None:
        fig.savefig(f"{file_path}{dataset_name[0:5]}_{window_size}_sliding_window.pdf")
        plt.close()
    else:
        plt.show()


def plot_memberwise_pathways(seas5_memberwise, seas5_med_results, era5_med_results):

    range_dict = {'Total': 400, 'Direct': 400, 'Indirect': 120, 'ENSO → SPV': 3, 'SPV → Regime': 60}

    for pathway in seas5_memberwise['pathway']:
        for cluster in seas5_memberwise['cluster']:

            if (cluster != 'NAO-'):  # or (pathway != 'ENSO → SPV'):
                pass
            else:
                n_windows = seas5_memberwise.sizes['window']
                n_iterations = seas5_memberwise.sizes['iteration']
                windows = np.arange(n_windows)

                synthetic_dist = seas5_memberwise.sel(pathway=pathway, cluster=cluster)['mediation_coeffs'].values
                era5_med = era5_med_results.sel(pathway=pathway, cluster=cluster)['mediation_mean'].values
                ens_mean_med = seas5_med_results.sel(pathway=pathway, cluster=cluster)['mediation_mean'].values

                fig, ax = plt.subplots(figsize=(3.5, 3))

                vmax = range_dict[pathway.item()]
                vmin = -vmax
                y_bins = np.linspace(vmin, vmax, 100)
                ax.set_ylim(vmin, vmax)
                norm = Normalize(vmin=0, vmax=150)

                density_matrix = np.zeros((len(y_bins)-1, n_windows))

                for w in range(n_windows):
                    hist, _ = np.histogram(synthetic_dist[:, w], bins=y_bins)
                    density_matrix[:, w] = hist

                X, Y = np.meshgrid(np.arange(n_windows + 1), y_bins)

                im = ax.pcolormesh(X, Y, density_matrix, cmap='Blues', norm=norm, shading='flat')

                p2_5 = np.percentile(synthetic_dist, 2.5, axis=0)
                p50 = np.percentile(synthetic_dist, 50.0, axis=0)
                p97_5 = np.percentile(synthetic_dist, 97.5, axis=0)
                p5 = np.percentile(synthetic_dist, 5, axis=0)
                p95 = np.percentile(synthetic_dist, 95, axis=0)

                x_centers = windows + 0.5

                # ax.fill_between(x_centers, p2_5, p97_5, color='tab:blue', alpha=0.2) #, label='95% CI')

                # # ax.plot(x_centers, p5, color='tab:blue', linewidth=1, linestyle='--', alpha=0.3)
                # # ax.plot(x_centers, p95, color='tab:blue', linewidth=1, linestyle='--', alpha=0.3)

                ax.plot(x_centers, era5_med, color='red', linewidth=1, label='ERA5')
                ax.plot(x_centers, ens_mean_med, color='black', linewidth=1, label='SEAS5 Ensemble Mean')

                if pathway == 'ENSO → SPV':
                    ax.set_title(f"{pathway.values} Pathway", fontsize=10)
                else:
                    ax.set_title(f"{cluster.values} {pathway.values} Pathway", fontsize=10)

                ax.axhline(0, color='grey', linestyle='--', linewidth=1)

                windows = era5_med_results.window.values

                major_labels = ['01-01', '02-01', '03-01']
                minor_labels = ['12-15', '01-15', '02-15', '03-15']
                minor_positions = np.where(np.isin(windows, minor_labels))[0]
                major_positions = np.where(np.isin(windows, major_labels))[0]

                ax.set_xticks(major_positions)
                ax.set_xticklabels(major_labels, rotation=0, ha='right', fontsize=10)

                ax.tick_params(axis='y', which='major', labelsize=10)

                ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_positions))
                ax.xaxis.set_minor_formatter(mticker.FixedFormatter(minor_labels))
                ax.tick_params(axis='x', which='minor', length=5, labelsize=8, rotation=0)
                ax.tick_params(axis='x', which='major', length=12, pad=6)

                for label in ax.get_xminorticklabels():
                    label.set_ha('center')

                for label in ax.get_xmajorticklabels():
                    label.set_ha('center')

                ax.set_ylabel("Pathway Strength")

                ax.set_xlim(x_centers[0]-0.5, x_centers[-1]-0.5)

                dist_patch = mpatches.Patch(color=plt.cm.Blues(0.5), label='SEAS5 Memberwise Distribution')

                handles, labels = ax.get_legend_handles_labels()

                # handles.append(dist_patch)
                # labels.append('SEAS5 Memberwise Distribution')

                ax.legend(handles=handles, labels=labels, loc='upper right')
                ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=10)

                # fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.92) # full 7x3
                fig.subplots_adjust(left=0.18, right=0.98, bottom=0.24, top=0.92)  # mini 3.5x3
                # fig.savefig(f"{file_path}_{cluster.values}_2000mem_samp_{pathway.values}_mediation_21d_sliding_window.pdf")
                # plt.close(fig)
                plt.show()


def plot_mediation_window_heatmaps(
    dataset,
    cluster_names=None,
    cmap="PuOr_r",
    symmetric=True,
    file_path=None,
    dataset_name="TEST",
    window_size='days'
):
    """
    Plot mediation heatmaps for sliding-window mediation results.

    dataset must contain:
        mediation_mean(window, cluster, pathway)
        mediation_sig(window, cluster, pathway)
    """

    windows = dataset.window.values
    clusters = dataset.cluster.values
    pathways = dataset.pathway.values

    range_dict = {'Total': 400, 'Direct': 400, 'Indirect': 120, 'ENSO → SPV': 3, 'SPV → Regime': 60}

    for pathway in pathways:

        da = dataset["mediation_mean"].sel(pathway=pathway)
        data = da.transpose("cluster", "window").values
        sig = dataset["mediation_sig"].sel(pathway=pathway).transpose("cluster", "window").values

        if pathway in ['ENSO not → SPV']:
            vmax = 0.5
            vmin = -0.5
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            ticks = np.linspace(vmin, vmax, 5)

        else:
            vmax = range_dict[pathway]
            vmin = -vmax
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            ticks = np.linspace(vmin, vmax, 5)

        if pathway == 'ENSO → SPV':
            fig, ax = plt.subplots(figsize=(3.5, 1.2))

            data = data[0].reshape(1, -1)

            im = ax.imshow(
                data,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                interpolation='none'
                # vmin=vmin,
                # vmax=vmax,
            )

            # significance lines
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if sig[i, j]:
                        ax.plot(j, i, "_", c='k', markersize=2)

            if cluster_names is None:
                cluster_names = clusters

            ax.set_yticks([0])
            ax.set_yticklabels(['SPV'])

            divider = make_axes_locatable(ax)

            cax = divider.append_axes("bottom", size="60%", pad=0.4)
            fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)

        else:
            fig, ax = plt.subplots(figsize=(3.5, 1.4))

            im = ax.imshow(
                data,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                interpolation='none'
                # vmin=vmin,
                # vmax=vmax,
            )

            # significance lines
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if sig[i, j]:
                        ax.plot(j, i, marker="_", c='k', markersize=2)

            if cluster_names is None:
                cluster_names = clusters

            ax.set_yticks(np.arange(len(cluster_names)))
            ax.set_yticklabels(cluster_names, fontsize=10)

            fig.colorbar(im, ax=ax, pad=0.02, location='right', orientation='vertical', ticks=ticks)

        # # X axis
        # spacing = max(len(windows) // 6, 1)
        # ax.set_xticks(np.arange(0, len(windows), spacing))
        # ax.set_xticklabels(windows[::spacing], rotation=45)

        # ['12-01','01-01','']
        # ax.set_xticks(np.arange(0, len(windows), spacing))
        # ax.set_xticklabels(windows[::spacing], rotation=45)

        # ax.tick_params(axis='both', which='major', labelsize=10)

        major_labels = ['01-01', '02-01', '03-01']
        minor_labels = ['12-15', '01-15', '02-15', '03-15']
        minor_positions = np.where(np.isin(windows, minor_labels))[0]
        major_positions = np.where(np.isin(windows, major_labels))[0]

        ax.set_xticks(major_positions)
        ax.set_xticklabels(major_labels, rotation=0, ha='right', fontsize=10)

        ax.tick_params(axis='y', which='major', labelsize=10)

        ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_positions))
        ax.xaxis.set_minor_formatter(mticker.FixedFormatter(minor_labels))
        ax.tick_params(axis='x', which='minor', length=5, labelsize=8, rotation=0)
        ax.tick_params(axis='x', which='major', length=12, pad=6)

        for label in ax.get_xminorticklabels():
            label.set_ha('center')

        for label in ax.get_xmajorticklabels():
            label.set_ha('center')

        ax.set_title(f"{dataset_name} {pathway}", fontsize=10)

        if pathway == 'ENSO → SPV':
            fig.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.8)
        else:
            fig.subplots_adjust(left=0.15, right=0.99, bottom=0.26, top=0.85)

        if file_path is not None:

            safe_pathway = pathway.replace("$\\to$", "_").replace(" ", "_")

            fig.savefig(f"{file_path}{dataset_name[0:5]}_{safe_pathway}_mediation_{window_size}_sliding_window.pdf")
            plt.close()
        else:
            plt.show()
