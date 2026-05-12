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
from IPython.display import display, Markdown
import calendar
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


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


def set_smooth_data_boundary(ax, lons, lats, n_points=500, proj=ccrs.PlateCarree()):
    lon_min, lon_max = np.min(lons), np.max(lons)
    lat_min, lat_max = np.min(lats), np.max(lats)
    top_lons = np.linspace(lon_min, lon_max, n_points)
    top_lats = np.full_like(top_lons, lat_max)
    right_lats = np.linspace(lat_max, lat_min, n_points)
    right_lons = np.full_like(right_lats, lon_max)
    bottom_lons = np.linspace(lon_max, lon_min, n_points)
    bottom_lats = np.full_like(bottom_lons, lat_min)
    left_lats = np.linspace(lat_min, lat_max, n_points)
    left_lons = np.full_like(left_lats, lon_min)
    all_lons = np.concatenate([top_lons, right_lons, bottom_lons, left_lons])
    all_lats = np.concatenate([top_lats, right_lats, bottom_lats, left_lats])
    [line] = ax.plot(all_lons, all_lats, transform=proj, linewidth=0, alpha=0)
    tx_path = line._get_transformed_path()
    path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
    polygon = mpath.Path(path_in_data_coords.vertices)
    ax.set_boundary(polygon)


def ERA5_SEAS5_cluster_centres_plot(era5_clusters, seas5_clusters, cluster_names, era5_cluster_order, seas5_cluster_order, k, levels=False):
    era5_cluster_centres = era5_clusters['cluster_centres'].values
    seas5_cluster_centres = seas5_clusters['cluster_centres'].values

    lats = era5_clusters.latitude.values
    lons = era5_clusters.longitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    if k == 6:

        cluster_centres_fig = plt.figure(figsize=(7, 2.2))
        gs = gridspec.GridSpec(2, 6, wspace=0.05, hspace=0.05, figure=cluster_centres_fig)
        # cluster_centres_fig.set_constrained_layout_pads(w_pad=1.0, h_pad=1.0, wspace=0.05, hspace=0.05)
        max_anomaly = max([era5_cluster_centres.max(), abs(era5_cluster_centres.min()), seas5_cluster_centres.max(), abs(seas5_cluster_centres.min())])
        if levels == True:
            levels = np.linspace(-max_anomaly, max_anomaly, 21)
        else:
            levels = 21
        for i in range(6):
            ax = cluster_centres_fig.add_subplot(gs[0, i], projection=ccrs.Orthographic(central_longitude=-30, central_latitude=45))
            ax.set_extent([-90, 30, 20, 80], crs=ccrs.PlateCarree())
            set_smooth_data_boundary(ax, lons, lats)
            ax.contourf(lon_grid, lat_grid, era5_cluster_centres[era5_cluster_order[i]], levels=levels, cmap='RdYlBu_r', transform=ccrs.PlateCarree())
            ax.set_title(f"{cluster_names[i]}", fontsize=11)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.7)

            ax1 = cluster_centres_fig.add_subplot(gs[1, i], projection=ccrs.Orthographic(central_longitude=-30, central_latitude=45))
            ax1.set_extent([-90, 30, 20, 80], crs=ccrs.PlateCarree())
            set_smooth_data_boundary(ax1, lons, lats)
            ax1.contourf(lon_grid, lat_grid, seas5_cluster_centres[seas5_cluster_order[i]], levels=levels, cmap='RdYlBu_r', transform=ccrs.PlateCarree())
            ax1.set_title(f"{cluster_names[i]}", fontsize=11)
            ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.7)

            if i == 0:
                ax.text(-0.2, 0.6, "ERA5", transform=ax.transAxes, fontsize=11, verticalalignment='center', rotation='vertical')
                ax1.text(-0.2, 0.6, "SEAS5", transform=ax1.transAxes, fontsize=11, verticalalignment='center', rotation='vertical')

    if k == 4:
        cluster_centres_fig = plt.figure(figsize=(7, 3.6))
        gs = gridspec.GridSpec(2, 4, wspace=0.05, hspace=0.05, figure=cluster_centres_fig)
        # cluster_centres_fig.set_constrained_layout_pads(w_pad=1.0, h_pad=1.0, wspace=0.05, hspace=0.05)
        max_anomaly = max([era5_cluster_centres.max(), abs(era5_cluster_centres.min()), seas5_cluster_centres.max(), abs(seas5_cluster_centres.min())])
        if levels == True:
            levels = np.linspace(-max_anomaly, max_anomaly, 21)
        else:
            levels = 21
        for i in range(4):
            ax = cluster_centres_fig.add_subplot(gs[0, i], projection=ccrs.Orthographic(central_longitude=-30, central_latitude=45))
            ax.set_extent([-90, 30, 20, 80], crs=ccrs.PlateCarree())
            set_smooth_data_boundary(ax, lons, lats)
            ax.contourf(lon_grid, lat_grid, era5_cluster_centres[era5_cluster_order[i]], levels=levels, cmap='RdYlBu_r', transform=ccrs.PlateCarree())
            ax.set_title(f"{cluster_names[i]}", fontsize=11)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.7)

            ax1 = cluster_centres_fig.add_subplot(gs[1, i], projection=ccrs.Orthographic(central_longitude=-30, central_latitude=45))
            ax1.set_extent([-90, 30, 20, 80], crs=ccrs.PlateCarree())
            set_smooth_data_boundary(ax1, lons, lats)
            ax1.contourf(lon_grid, lat_grid, seas5_cluster_centres[seas5_cluster_order[i]], levels=levels, cmap='RdYlBu_r', transform=ccrs.PlateCarree())
            ax1.set_title(f"{cluster_names[i]}", fontsize=11)
            ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.7)

            if i == 0:
                ax.text(-0.2, 0.6, "ERA5", transform=ax.transAxes, fontsize=11, verticalalignment='center', rotation='vertical')
                ax1.text(-0.2, 0.6, "SEAS5", transform=ax1.transAxes, fontsize=11, verticalalignment='center', rotation='vertical')
    return cluster_centres_fig


def plot_cluster_centroids(era5_clusters, cluster_names, cluster_order=None, levels=None, title=None):

    data = era5_clusters['cluster_centres'].values
    lats = era5_clusters.latitude.values
    lons = era5_clusters.longitude.values

    k = data.shape[0]

    if cluster_order is None:
        cluster_order = np.arange(k)

    ncols = 3 if k <= 3 else 4
    if k == 6:
        ncols = 2
    nrows = int(np.ceil(k / ncols))

    fig = plt.figure(figsize=(5, 5.8))
    height_ratios = [1] * nrows + [0.12]
    gs = gridspec.GridSpec(nrows+1, ncols, wspace=0.1, hspace=0.2, height_ratios=height_ratios)

    if levels is None:
        max_val = max(abs(data.min()), abs(data.max()))
        levels = np.linspace(-max_val, max_val, 21)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    for i in range(k):
        cluster_idx = cluster_order[i]

        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col], projection=ccrs.Orthographic(central_longitude=-30, central_latitude=45))

        ax.set_extent([-90, 30, 20, 80], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, alpha=0.8)
        set_smooth_data_boundary(ax, lons, lats)

        centroid = data[cluster_idx]
        cf = ax.contourf(lon_grid, lat_grid, centroid,
                         levels=levels, cmap='RdYlBu_r',
                         transform=ccrs.PlateCarree(), extend='neither')

        # ax.set_title(f"{cluster_names[i]}", fontsize=14, y=1)
        ax.text(0.2, 0.85, cluster_names[i], transform=ax.transAxes, va='center', ha='right', fontsize=14, weight='bold', clip_on=False)

    if title:
        fig.suptitle(f'{title}', fontsize=18, y=0.9)
    cax = fig.add_subplot(gs[-1, :])

    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, cax_pos.y0+0.1, cax_pos.width, cax_pos.height])
    colorbar = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=[-250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250], extend='neither')
    colorbar.set_label('Geopotential Height Anomaly (gpm)', size=10)
    colorbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.97)

    return fig


def plot_ensemble_ssw_timing_skill(data, p_values, lower_quantiles, upper_quantiles, metric, pathway=None):
    """
    Plot correlations for a single pathway (e.g., ENSO -> SPV) with no regimes.
    Layout: Joint-plot style (Fixed alignment)
    - Top Right (Main): Transition period (2D heatmap square)
    - Center Left: Early period (1D array, vertical, shares Y with Transition)
    - Bottom Right: Late period (1D array, horizontal, shares X with Transition)
    """
    n_lower = len(lower_quantiles)
    n_upper = len(upper_quantiles)

    fig, ax3 = plt.subplots(figsize=(3.5, 2.8))

    if metric == 'slope':
        vmax, vmin = 3.0, -1.0
        cmap = 'PiYG'
        ticks = [-1, 0, 1, 2, 3]
    elif metric == 'correlation':
        vmax, vmin = 0.5, -0.5
        cmap = 'RdBu_r'
        ticks = [-0.5, -0.25, 0, 0.25, 0.5]
    elif metric == 'teleconnection':
        vmax = np.ceil(np.max(np.abs(data.values)))
        vmax, vmin = vmax, -vmax
        cmap = 'PuOr_r'
        # ticks = [vmin, vmin//2, 0, vmax//2, vmax]
    else:
        max_val = np.max([data.max().item(), np.abs(data.min().item())])
        vmax, vmin = max_val, -max_val
        cmap = 'RdBu_r'

    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad(color='gray')
    lower_cell_width = (lower_quantiles[-1] - lower_quantiles[0]) / (n_lower - 1) if n_lower > 1 else 0.1
    upper_cell_width = (upper_quantiles[-1] - upper_quantiles[0]) / (n_upper - 1) if n_upper > 1 else 0.1

    early_data, early_sig = data.sel(period='Early').mean(dim='upper_q').values, p_values.sel(period='Early').mean(dim='upper_q').values < 0.05
    trans_data, trans_sig = data.sel(period='Transition').values, p_values.sel(period='Transition').values < 0.05
    late_data, late_sig = data.sel(period='Late').mean(dim='lower_q').values, p_values.sel(period='Late').mean(dim='lower_q').values < 0.05
    E_T_data, E_T_sig = data.sel(period='Early+Transition').mean(dim='lower_q').values, p_values.sel(period='Early+Transition').mean(dim='lower_q').values < 0.05
    T_L_data, T_L_sig = data.sel(period='Transition+Late').mean(dim='upper_q').values, p_values.sel(period='Transition+Late').mean(dim='upper_q').values < 0.05

    im3 = ax3.imshow(trans_data, aspect='equal', cmap=cmap,
                     extent=[upper_quantiles[0] - upper_cell_width/2, upper_quantiles[-1] + upper_cell_width/2,
                             lower_quantiles[-1] + lower_cell_width/2, lower_quantiles[0] - lower_cell_width/2],
                     origin='upper', vmin=vmin, vmax=vmax)

    ax3.tick_params(labelbottom=False, labelleft=False)

    # X, Y = np.meshgrid(upper_quantiles, lower_quantiles)

    mpl.rc('hatch', color='k', linewidth=0.5)

    # ax3.contourf(
    #     X,
    #     Y,
    #     trans_sig,
    #     levels=[0.5, 1],
    #     colors='none',
    #     hatches=['////'],
    #     alpha=0
    # )

    x_edges = np.concatenate([
        upper_quantiles - upper_cell_width/2,
        [upper_quantiles[-1] + upper_cell_width/2]
    ])

    y_edges = np.concatenate([
        lower_quantiles - lower_cell_width/2,
        [lower_quantiles[-1] + lower_cell_width/2]
    ])

    # imshow is flipped vertically
    y_edges = y_edges[::-1]
    sig_mask = trans_sig[::-1, :]

    sig_masked = np.ma.masked_where(~sig_mask, sig_mask)  # Mask OUT nonsignificant cells

    # transparent colormap so only hatching is drawn
    none_cmap = ListedColormap(['none'])

    mpl.rcParams['hatch.linewidth'] = 0.5

    ax3.pcolor(
        x_edges,
        y_edges,
        sig_masked,
        cmap=none_cmap,
        hatch='////',
        edgecolor='k',   # hatch color comes from edgecolor
        linewidth=0,
        zorder=10
    )

    divider = make_axes_locatable(ax3)

    ax1 = divider.append_axes("left", size="10%", pad=0.15, sharey=ax3)  # Early

    im1 = ax1.imshow(early_data[:, np.newaxis], aspect='auto', cmap=cmap,
                     extent=[-0.5, 0.5, lower_quantiles[-1] + lower_cell_width/2, lower_quantiles[0] - lower_cell_width/2],
                     vmin=vmin, vmax=vmax)

    ax1.set_ylabel('')
    ax1.set_yticks([0.5, 0.25, 0.0])
    ax1.set_xticks([])

    ax1.tick_params(labelbottom=False, labelleft=False)

    ax4 = divider.append_axes("left", size="10%", pad=0.25, sharey=ax1)  # T+L

    im4 = ax4.imshow(T_L_data[:, np.newaxis], aspect='auto', cmap=cmap,
                     extent=[-0.5, 0.5, lower_quantiles[-1] + lower_cell_width/2, lower_quantiles[0] - lower_cell_width/2],
                     vmin=vmin, vmax=vmax)

    ax4.set_ylabel('Lower SSW Timing Percentile')
    ax4.set_yticks([0.5, 0.25, 0.0])
    ax4.set_xticks([])

    ax2 = divider.append_axes("bottom", size="10%", pad=0.1, sharex=ax3)  # Late

    im2 = ax2.imshow(late_data[np.newaxis, :], aspect='auto', cmap=cmap,
                     extent=[upper_quantiles[0] - upper_cell_width/2, upper_quantiles[-1] + upper_cell_width/2,
                             0.5, -0.5],
                     vmin=vmin, vmax=vmax)

    ax2.set_xlabel('')
    ax2.set_xticks([0.5, 0.25, 0.0])
    ax2.set_yticks([])
    ax2.tick_params(labelbottom=False, labelleft=False)

    ax5 = divider.append_axes("bottom", size="10%", pad=0.1, sharex=ax2)  # E+T

    im5 = ax5.imshow(E_T_data[np.newaxis, :], aspect='auto', cmap=cmap,
                     extent=[upper_quantiles[0] - upper_cell_width/2, upper_quantiles[-1] + upper_cell_width/2,
                             0.5, -0.5],
                     vmin=vmin, vmax=vmax)

    ax5.set_xlabel('Upper SSW Timing Percentile')
    ax5.set_xticks([0.5, 0.75, 1.0])
    ax5.set_yticks([])

    def add_sig_bar(ax, sig_mask, coords, cell_width, y=0, vertical=False):
        start = None
        for i, is_sig in enumerate(sig_mask):
            if is_sig and start is None:
                start = i
            is_last = (i == len(sig_mask) - 1)
            if start is not None and ((not is_sig) or is_last):
                end = i if (is_sig and is_last) else i - 1
                if vertical:
                    ax.vlines(
                        y,
                        coords[start] - cell_width/2,
                        coords[end] + cell_width/2,
                        color='k',
                        linewidth=0.5,
                        zorder=10
                    )
                else:
                    ax.hlines(
                        y,
                        coords[start] - cell_width/2,
                        coords[end] + cell_width/2,
                        color='k',
                        linewidth=0.5,
                        zorder=10
                    )
                start = None

    add_sig_bar(ax1, early_sig, lower_quantiles, lower_cell_width, vertical=True)
    add_sig_bar(ax2, late_sig, upper_quantiles, upper_cell_width)
    add_sig_bar(ax4, T_L_sig, lower_quantiles, lower_cell_width, vertical=True)
    add_sig_bar(ax5, E_T_sig, upper_quantiles, upper_cell_width)

    # ax2.yaxis.set_label_position("right")
    # ax5.yaxis.set_label_position("right")
    ax1.set_title('Early', fontsize=10, pad=5)
    ax2.set_ylabel('Late', rotation=0, labelpad=5, va='center', ha='right', fontsize=10)
    ax3.set_title('Transition', pad=5)
    ax4.set_title('T+L', fontsize=10, pad=5)
    ax5.set_ylabel('E+T', rotation=0, labelpad=5, va='center', ha='right', fontsize=10)

    if 'ticks' in locals():
        # Appending colorbar to divider guarantees it aligns cleanly with the main plot's height too
        cbar_ax = divider.append_axes("right", size="5%", pad=0.15)
        colorbar = fig.colorbar(im3, cax=cbar_ax, ticks=ticks)
        if metric == 'slope':
            colorbar.set_label('Regression Slope', size=10)
        elif metric == 'teleconnection':
            colorbar.set_label(f'{pathway} Strength', size=10)
        else:
            colorbar.set_label('Correlation', size=10)

    fig.subplots_adjust(
        left=0.18,
        right=0.8,
        bottom=0,
        top=1.1
    )

    print('Early Max', np.round(np.nanmax(early_data), 2))
    print('Transition Max', np.round(np.nanmax(trans_data), 2))
    print('Late Max', np.round(np.nanmax(late_data), 2))
    print('Early+Transition Max', np.round(np.nanmax(E_T_data), 2))
    print('Transition+Late Max', np.round(np.nanmax(T_L_data), 2))

    print('-'*40)

    print('Early Mean', np.round(np.nanmean(early_data[early_sig]), 2))
    print('Transition Mean', np.round(np.nanmean(trans_data[trans_sig]), 2))
    print('Late Mean', np.round(np.nanmean(late_data), 2))
    print('Early+Transition Mean', np.round(np.nanmean(E_T_data[E_T_sig]), 2))
    print('Transition+Late Mean', np.round(np.nanmean(T_L_data[T_L_sig]), 2))

    return fig


def main():
    print('Ooops')


if __name__ == "__main__":
    start_time: float = time.time()

    main()

    end_time: float = time.time()
    print(f'Time taken to complete code:{end_time-start_time}')
