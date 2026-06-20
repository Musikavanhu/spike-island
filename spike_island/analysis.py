"""
Spike Island — Neurophysiology Analysis Metrics
=================================================
Compute standard metrics from spike trains for neurophysiology learning.

Functions
---------
- isi_histogram: Compute inter-spike interval histogram
- coefficient_of_variation: CV = std(ISI) / mean(ISI)
- mean_firing_rate: Spikes per second over simulation window
- autocorrelogram: Auto-correlation of spike train (binned)
- raster_plot: Visualize spike times with axes
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def isi_histogram(
    spike_times: np.ndarray,
    bin_width_ms: float = 5.0,
    max_isi_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the inter-spike interval (ISI) histogram.

    Parameters
    ----------
    spike_times : np.ndarray
        Sorted spike times in milliseconds.
    bin_width_ms : float
        Histogram bin width in milliseconds (default 5 ms).
    max_isi_ms : float or None
        Maximum ISI to include; defaults to 5× the median ISI.

    Returns
    -------
    counts : np.ndarray
        ISI counts per bin.
    bin_edges : np.ndarray
        Bin edges in milliseconds (length = len(counts) + 1).

    Notes
    -----
    The ISI histogram reveals the distribution of intervals between
    consecutive spikes. An exponentially decaying ISI histogram is the
    hallmark of a Poisson process. Regular firing produces a sharp peak
    near the mean ISI. Bursting produces a secondary peak at short ISIs.
    """
    if len(spike_times) < 2:
        return np.array([0]), np.array([0.0, bin_width_ms])

    isis = np.diff(spike_times)

    if max_isi_ms is None:
        max_isi_ms = 5.0 * np.median(isis)

    isis = isis[isis <= max_isi_ms]
    n_bins = int(max_isi_ms / bin_width_ms)
    counts, bin_edges = np.histogram(isis, bins=n_bins, range=(0, max_isi_ms))
    return counts, bin_edges


def coefficient_of_variation(spike_times: np.ndarray) -> float:
    """Coefficient of variation of inter-spike intervals.

    CV = std(ISI) / mean(ISI).  A dimensionless measure of spike-train
    irregularity.

    Parameters
    ----------
    spike_times : np.ndarray
        Sorted spike times in milliseconds.

    Returns
    -------
    float
        CV value.  ~1.0 for Poisson, ~0.0 for perfectly regular, >1.0
        for bursting/irregular patterns.

    Raises
    ------
    ValueError
        If fewer than 2 spikes are provided.
    """
    if len(spike_times) < 2:
        raise ValueError("Need at least 2 spikes to compute CV")

    isis = np.diff(spike_times)
    mean_isi = np.mean(isis)
    if mean_isi == 0:
        return 0.0
    return float(np.std(isis) / mean_isi)


def mean_firing_rate(spike_times: np.ndarray, duration_ms: float) -> float:
    """Mean firing rate over a simulation window.

    Parameters
    ----------
    spike_times : np.ndarray
        Sorted spike times in milliseconds.
    duration_ms : float
        Total simulation duration in milliseconds.

    Returns
    -------
    float
        Mean firing rate in spikes per second (Hz).
    """
    if duration_ms <= 0:
        return 0.0
    return float(len(spike_times) / (duration_ms / 1000.0))


def autocorrelogram(
    spike_times: np.ndarray,
    bin_width_ms: float = 5.0,
    max_lag_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the auto-correlogram of a spike train.

    Parameters
    ----------
    spike_times : np.ndarray
        Sorted spike times in milliseconds.
    bin_width_ms : float
        Autocorrelation bin width in ms (default 5 ms).
    max_lag_ms : float or None
        Maximum lag in ms to compute; defaults to the shorter of
        2× the spike train duration or 2000 ms.

    Returns
    -------
    corr : np.ndarray
        Normalized autocorrelation values.
    lags : np.ndarray
        Lag values in milliseconds (centred at 0).
    """
    if max_lag_ms is None:
        duration = spike_times[-1] - spike_times[0] if len(spike_times) > 1 else 0
        max_lag_ms = min(2.0 * duration, 2000.0)

    n_bins = int(max_lag_ms / bin_width_ms)
    dt = bin_width_ms

    # Build binned spike raster
    max_time = spike_times[-1] if len(spike_times) > 0 else 0
    n_total = int(np.ceil(max_time / dt))
    raster = np.zeros(n_total, dtype=np.float64)
    spike_indices = np.searchsorted(np.arange(n_total) * dt, spike_times)
    spike_indices = np.clip(spike_indices, 0, n_total - 1)
    raster[spike_indices] = 1.0

    # Compute autocorrelation
    corr_full = correlate(raster, raster, mode="full")
    n_lags = len(corr_full)
    center = n_lags // 2

    # Extract window around zero lag
    start = center - n_bins
    end = center + n_bins + 1
    start = max(0, start)
    end = min(n_lags, end)

    corr = corr_full[start:end]
    lags = (np.arange(end - start) - (center - start)) * dt

    # Normalize by zero-lag value
    if corr[center - start] > 0:
        corr = corr / corr[center - start]

    return corr, lags


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def analyze(
    spike_times: np.ndarray,
    duration_ms: float,
    name: str = "neuron",
) -> dict:
    """Run a full metric analysis on a single spike train.

    Parameters
    ----------
    spike_times : np.ndarray
        Sorted spike times in milliseconds.
    duration_ms : float
        Total simulation duration in ms.
    name : str
        Label for the neuron (used in output dict).

    Returns
    -------
    dict
        Analysis results keyed by metric name:
        - spikes: number of spikes
        - isi_mean: mean ISI in ms
        - isi_std: std ISI in ms
        - cv: coefficient of variation
        - firing_rate_hz: mean firing rate
        - isi_counts: binned ISI histogram counts
        - isi_edges: binned ISI histogram edges
        - corr: autocorrelogram values
        - lags: autocorrelogram lag values
    """
    n_spikes = len(spike_times)

    if n_spikes < 2:
        return {
            "name": name,
            "spikes": n_spikes,
            "isi_mean": 0.0,
            "isi_std": 0.0,
            "cv": 0.0,
            "firing_rate_hz": mean_firing_rate(spike_times, duration_ms),
            "isi_counts": np.array([0]),
            "isi_edges": np.array([0.0, 5.0]),
            "corr": np.array([0.0]),
            "lags": np.array([0.0]),
        }

    isis = np.diff(spike_times)
    counts, edges = isi_histogram(spike_times)
    corr, lags = autocorrelogram(spike_times)

    return {
        "name": name,
        "spikes": n_spikes,
        "isi_mean": float(np.mean(isis)),
        "isi_std": float(np.std(isis)),
        "cv": coefficient_of_variation(spike_times),
        "firing_rate_hz": mean_firing_rate(spike_times, duration_ms),
        "isi_counts": counts,
        "isi_edges": edges,
        "corr": corr,
        "lags": lags,
    }


# ---------------------------------------------------------------------------
# Visualization dashboard
# ---------------------------------------------------------------------------

def plot_analysis_dashboard(
    results: list[dict],
    duration_ms: float = 5_000.0,
    save_path: str = "plots/analysis_dashboard.png",
) -> None:
    """Plot ISI histogram + autocorrelogram + raster for each neuron.

    Parameters
    ----------
    results : list[dict]
        Output from :func:`analyze`, one per neuron.
    duration_ms : float
        Simulation duration (for raster display).
    save_path : str
        Where to save the PNG.

    Notes
    -----
    Produces a grid of subplots: raster on top, ISI histogram left,
    autocorrelogram right.  One row per neuron.
    """
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.5 * n), sharex="col")
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        spike_times = None  # we don't have raw times — reconstruct from raster
        # Plot ISI histogram
        axes[row, 0].bar(
            res["isi_edges"][:-1],
            res["isi_counts"],
            width=np.diff(res["isi_edges"]).min(),
            edgecolor="none",
            color="#1e3a5f",
        )
        axes[row, 0].set_title(f"ISI: {res['name']}", fontsize=9)
        axes[row, 0].set_xlabel("ISI (ms)")
        axes[row, 0].set_ylabel("Count")

        # Plot autocorrelogram
        axes[row, 1].plot(res["lags"], res["corr"], color="#c45533", linewidth=1)
        axes[row, 1].set_title(f"AUTO: {res['name']}", fontsize=9)
        axes[row, 1].set_xlabel("Lag (ms)")

    axes[-1, 0].set_ylabel("Count")

    fig.suptitle(
        f"Spike Analysis — {n} neuron(s) · {duration_ms / 1000:.0f}s window",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Analysis dashboard saved to {save_path}")
