import xskillscore as xs
import xarray as xr
import numpy as np
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
import plotting as pl
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy import stats
from sklearn.cluster import KMeans
import anomaly_and_clustering as ac
import sklearn as sk
import xeofs as xe
import time


def get_ssw_boundaries(seas5_u10_da):

    is_easterly = seas5_u10_da < 0
    is_westerly = seas5_u10_da > 0

    # the onset must be within NDJFM but the recovery is before end of april so need a mask for each
    is_ndjfm = seas5_u10_da.valid_time.dt.month.isin([11, 12, 1, 2, 3])
    before_may = seas5_u10_da.valid_time.dt.month <= 4

    # Recovery = 10 consecutive westerly days before end of April
    recovery_window = 10

    sustained_westerly = (
        is_westerly
        .rolling(step=recovery_window)
        .min()
        .fillna(0)
        .astype(bool)
    )

    recovery_before_may = sustained_westerly & before_may

    has_future_recovery = (
        recovery_before_may
        .isel(step=slice(None, None, -1))
        .cumsum(dim='step')
        .isel(step=slice(None, None, -1))
    ) >= 1

    # Candidate onset (only NDJFM)
    ssw_min_duration = 1

    sustained_easterly = (
        is_easterly
        .isel(step=slice(None, None, -1))
        .rolling(step=ssw_min_duration)
        .min()
        .fillna(0)
        .astype(bool)
        .isel(step=slice(None, None, -1))
    )

    candidate_onsets = sustained_easterly & is_ndjfm

    # Valid SSW = onset in NDJFM + recovery before May
    valid_ssw = candidate_onsets & has_future_recovery

    first_ssw_step = seas5_u10_da.step.where(valid_ssw).min(dim='step')

    return first_ssw_step


def ssw_timing_figure(onset_steps, all_steps, years, method='dynamic', lq=0.25, uq=0.75):
    if method == 'clim':
        lq_step = onset_steps.quantile(lq, dim=["time", "number"])
        uq_step = onset_steps.quantile(uq, dim=["time", "number"])
    else:
        lq_step = onset_steps.quantile(lq, dim="number")
        uq_step = onset_steps.quantile(uq, dim="number")

    start_step = all_steps.min().values
    end_step = all_steps.max().values

    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # 3.5inch is half the width of an a4 page with 0.5inch margines. 7.3 would be the full width
    c_early = 'tab:green'
    c_trans = 'tab:red'
    c_late = 'tab:cyan'

    y_pos = np.arange(len(years))

    ax.barh(y_pos, width=(lq_step - start_step), left=start_step,
            color=c_early, height=1, edgecolor='white', linewidth=0.5)

    ax.barh(y_pos, width=(uq_step - lq_step), left=lq_step,
            color=c_trans, height=1, edgecolor='white', linewidth=0.5)

    ax.barh(y_pos, width=(end_step - uq_step), left=uq_step,
            color=c_late, height=1, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos[::5])
    ax.set_yticklabels(years[::5])

    ax.set_ylim(y_pos[0]-1, y_pos[-1]+1)

    tick_locs = [720, 1464, 2208, 2880, 3600]
    tick_labs = ['12-01', '01-01', '02-01', '03-01', '03-31']
    ax.set_xlim(720, 3600)

    valid_ticks = [(loc, lab) for loc, lab in zip(tick_locs, tick_labs)
                   if loc <= end_step]

    if valid_ticks:
        ax.set_xticks([x[0] for x in valid_ticks])
        ax.set_xticklabels([x[1] for x in valid_ticks])

    ax.tick_params(axis='both', which='major')

    legend_elements = [
        Patch(facecolor=c_early, label='Early'),
        Patch(facecolor=c_trans, label='Transition'),
        Patch(facecolor=c_late,  label='Late')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', c='white')

    fig.subplots_adjust(top=0.98, bottom=0.12)

    return fig


def compute_ssw_boundaries_grid(first_ssw, lower_qs, upper_qs):
    q_all = np.unique(np.concatenate([lower_qs, upper_qs]))
    q_values = first_ssw.quantile(q_all, dim="number")

    # FIX 1: Rename the dimensions so xarray broadcasts them into a 2D grid
    early = q_values.sel(quantile=lower_qs).rename({"quantile": "lower_q"})
    late = q_values.sel(quantile=upper_qs).rename({"quantile": "upper_q"})

    return xr.Dataset({
        "early": early,
        "late": late
    })


def build_period_masks(boundaries, step, start=720, end=3600):
    early = boundaries["early"]   # (time, lower_q)
    late = boundaries["late"]     # (time, upper_q)

    early, late = xr.broadcast(early, late)

    step = xr.DataArray(step, dims=["step"])

    # Xarray now broadcasts step, early, and late into shape: (step, time, lower_q, upper_q)
    masks = xr.Dataset({
        "Early": (step >= start) & (step <= early),
        "Transition": (step > early) & (step < late),
        "Late": (step >= late) & (step <= end),
    })

    masks["Early+Transition"] = masks["Early"] | masks["Transition"]
    masks["Transition+Late"] = masks["Transition"] | masks["Late"]
    masks["Early+Late"] = masks["Early"] | masks["Late"]

    return masks


def masked_mean(da, mask):
    if "quantile" in mask.dims and "quantile" not in da.dims:
        da = da.expand_dims(
            quantile=mask.coords["quantile"].values
        )

    da, mask = xr.align(da, mask, join="inner")

    return (da * mask).sum("step") / mask.sum("step")


def compute_period_features(data, masks, min_days=21):
    out = []

    for period in masks:
        mask = masks[period]
        period_length = mask.sum("step")
        is_valid_length = period_length >= min_days

        period_ds = {}

        for name, da in data.items():

            # .where(is_valid_length) turns the year to NaN if the period was shorter than min_days
            da_masked = da.where(mask).mean("step").where(is_valid_length)

            if "number" in da_masked.dims:
                da_masked = da_masked.mean("number")

            period_ds[name] = da_masked

        ds = xr.Dataset(period_ds).expand_dims(period=[period])
        out.append(ds)

    return xr.concat(out, dim="period")


def compute_skill(features, variable_pairs, min_samples=21):
    results = []

    for svar, evar in variable_pairs:
        s5 = features[svar]
        e5 = features[evar]

        s5, e5 = xr.align(s5, e5)

        valid = s5.notnull() & e5.notnull()
        count = valid.sum("time")
        mask = count >= min_samples

        r = xs.pearson_r(s5, e5, dim="time")
        rmse_model = xs.rmse(s5, e5, dim="time")

        clim = e5.mean("time")
        clim_ts = clim.broadcast_like(e5)
        rmse_clim = xs.rmse(clim_ts, e5, dim="time")

        skill = 1 - rmse_model / rmse_clim

        ds = xr.Dataset({
            "correlation": r.where(mask),
            "skill": skill.where(mask)
        })

        results.append(ds.assign_coords(variable=svar))

    return xr.concat(results, dim="variable")


def compute_mediation(features, x, m, y):

    x = features[x]
    m = features[m]
    y = features[y]

    x, m, y = xr.align(x, m, y)

    a = xs.linslope(x, m, dim="time")
    b = xs.linslope(m, y, dim="time")
    c = xs.linslope(x, y, dim="time")

    indirect = a * b
    direct = c - indirect

    return xr.Dataset({
        "a": a,
        "b": b,
        "c": c,
        "indirect": indirect,
        "direct": direct
    })
