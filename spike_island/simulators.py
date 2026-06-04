"""
Spike Island — Spike Train Simulators
======================================
Generate synthetic neural spike trains with different firing patterns.

Each generator returns spike times as a sorted numpy array (in milliseconds).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sort_spikes(spike_times: np.ndarray) -> np.ndarray:
    """Return spike times sorted in ascending order."""
    return np.sort(spike_times)


# ---------------------------------------------------------------------------
# Poisson spike train
# ---------------------------------------------------------------------------

def poisson_spikes(
    rate_hz: float,
    duration_ms: float = 10_000.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a spike train from a homogeneous Poisson process.

    Parameters
    ----------
    rate_hz : float
        Mean firing rate in spikes per second.
    duration_ms : float
        Total simulation duration in milliseconds (default 10 s).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted spike times in milliseconds.

    Notes
    -----
    A Poisson process is memoryless: each small time interval Δt has
    probability `rate_hz * Δt / 1000` of containing a spike, independent
    of everything else. This is the simplest model of neuronal spiking
    and serves as a baseline for more complex patterns.
    """
    rng = np.random.default_rng(seed)
    dt_ms = 1.0  # 1 ms bins
    n_bins = int(duration_ms / dt_ms)
    p = rate_hz * dt_ms / 1000.0
    spikes = rng.binomial(1, p, size=n_bins)
    spike_times = np.flatnonzero(spikes).astype(np.float64) * dt_ms
    return spike_times


# ---------------------------------------------------------------------------
# Refractory Poisson (absolute refractory period)
# ---------------------------------------------------------------------------

def refractory_poisson(
    rate_hz: float,
    duration_ms: float = 10_000.0,
    refractory_ms: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a spike train with an absolute refractory period.

    After each spike, the neuron cannot fire again for `refractory_ms`
    milliseconds. This is a more biologically realistic Poisson process.

    Parameters
    ----------
    rate_hz : float
        Target mean firing rate (actual rate will be slightly lower
        due to the refractory period).
    duration_ms : float
        Total simulation duration in milliseconds.
    refractory_ms : float
        Absolute refractory period in milliseconds (default 2 ms).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted spike times in milliseconds.
    """
    rng = np.random.default_rng(seed)
    spike_times: list[float] = []
    last_spike_time = -refractory_ms - 1.0  # ensure first spike can fire
    dt_ms = 1.0
    p = rate_hz * dt_ms / 1000.0
    t = 0.0

    while t < duration_ms:
        if t - last_spike_time >= refractory_ms:
            if rng.random() < p:
                spike_times.append(t)
                last_spike_time = t
        t += dt_ms

    return np.array(spike_times, dtype=np.float64)


# ---------------------------------------------------------------------------
# Bursty spike train (Poisson + bursts)
# ---------------------------------------------------------------------------

def bursty_poisson(
    background_rate_hz: float,
    burst_rate: float = 5.0,
    burst_size_mean: int = 4,
    burst_size_std: float = 1.5,
    intra_burst_isi_ms: float = 5.0,
    duration_ms: float = 10_000.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a spike train with Poisson background + burst events.

    Bursts occur as a separate Poisson process at `burst_rate` events/sec.
    Each burst contains a random number of spikes with near-zero ISI
    (controlled by `intra_burst_isi_ms`).

    Parameters
    ----------
    background_rate_hz : float
        Background Poisson firing rate.
    burst_rate : float
        Rate of burst events per second (default 5 Hz).
    burst_size_mean : int
        Mean number of spikes per burst.
    burst_size_std : float
        Standard deviation of burst size.
    intra_burst_isi_ms : float
        Mean ISI within a burst in milliseconds.
    duration_ms : float
        Total simulation duration in milliseconds.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted spike times in milliseconds.

    Notes
    -----
    Bursting is a common neuronal firing pattern associated with
    information coding in many brain regions. High-frequency bursts
    within a spike train can signal different things than the same
    number of spikes spread out over time.
    """
    rng = np.random.default_rng(seed)

    # Background spikes
    background = poisson_spikes(background_rate_hz, duration_ms, seed)

    # Burst events
    burst_times = poisson_spikes(burst_rate, duration_ms, seed)

    # Generate intra-burst spikes for each burst event
    burst_spikes: list[float] = []
    for bt in burst_times:
        n_spikes = max(1, int(rng.normal(burst_size_mean, burst_size_std)))
        isis = np.abs(rng.normal(intra_burst_isi_ms, intra_burst_isi_ms * 0.3, size=n_spikes - 1))
        burst_spikes.append(bt)
        burst_spikes.extend(bt + np.cumsum(isis))

    all_spikes = np.concatenate([background, burst_spikes])
    return _sort_spikes(all_spikes)


# ---------------------------------------------------------------------------
# Rhythmic spike train (regular intervals with jitter)
# ---------------------------------------------------------------------------

def rhythmic_spikes(
    rate_hz: float,
    duration_ms: float = 10_000.0,
    jitter_sd_ms: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a rhythmic spike train with timing jitter.

    Spikes occur at regular intervals corresponding to the given rate,
    but with Gaussian jitter added to simulate biological variability.

    Parameters
    ----------
    rate_hz : float
        Target firing rate in Hz.
    duration_ms : float
        Total simulation duration in milliseconds.
    jitter_sd_ms : float
        Standard deviation of timing jitter in milliseconds (default 2 ms).
        Set to 0 for perfectly regular spiking.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted spike times in milliseconds.

    Notes
    -----
    Regular firing patterns are often associated with pacemaker neurons
    and can carry different information than irregular firing. The
    coefficient of variation (CV = std(ISI)/mean(ISI)) near 0 indicates
    high regularity.
    """
    rng = np.random.default_rng(seed)
    isi_ms = 1000.0 / rate_hz
    n_spikes = int(duration_ms / isi_ms)
    jitter = rng.normal(0, jitter_sd_ms, size=n_spikes)
    spike_times = np.cumsum(np.maximum(1.0, isi_ms + jitter))
    spike_times = spike_times[spike_times < duration_ms]
    return spike_times


# ---------------------------------------------------------------------------
# Main — quick demo
# ---------------------------------------------------------------------------

def demo() -> None:
    """Run a quick demo of all spike train generators."""
    import matplotlib.pyplot as plt

    rate = 20.0  # Hz
    duration = 5_000.0  # 5 seconds

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    titles = [
        "Poisson (rate=20 Hz)",
        "Refractory Poisson (refractory=2 ms)",
        "Bursty Poisson (burst rate=5 Hz)",
        "Rhythmic (rate=10 Hz, jitter=1 ms)",
        "Rhythmic (rate=10 Hz, jitter=5 ms)",
    ]

    axes[0].eventplot(
        poisson_spikes(rate, duration, seed=42),
        linelengths=0.8,
        colors="black",
    )
    axes[0].set_title(titles[0], fontsize=10)

    axes[1].eventplot(
        refractory_poisson(rate, duration, refractory_ms=2.0, seed=42),
        linelengths=0.8,
        colors="black",
    )
    axes[1].set_title(titles[1], fontsize=10)

    axes[2].eventplot(
        bursty_poisson(rate / 2, 5.0, 4, 1.5, 5.0, duration, seed=42),
        linelengths=0.8,
        colors="black",
    )
    axes[2].set_title(titles[2], fontsize=10)

    axes[3].eventplot(
        rhythmic_spikes(10.0, duration, jitter_sd_ms=1.0, seed=42),
        linelengths=0.8,
        colors="black",
    )
    axes[3].set_title(titles[3], fontsize=10)

    axes[4].eventplot(
        rhythmic_spikes(10.0, duration, jitter_sd_ms=5.0, seed=42),
        linelengths=0.8,
        colors="black",
    )
    axes[4].set_title(titles[4], fontsize=10)

    for ax in axes:
        ax.set_yticks([])
        ax.set_xlim(0, duration)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig("plots/spike_train_demo.png", dpi=150)
    print("Demo plot saved to plots/spike_train_demo.png")


if __name__ == "__main__":
    demo()
