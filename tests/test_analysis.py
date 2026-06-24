"""Tests for spike_island.analysis."""

from __future__ import annotations

import numpy as np
import pytest

from spike_island.analysis import (
    analyze,
    autocorrelogram,
    coefficient_of_variation,
    isi_histogram,
    mean_firing_rate,
)


# ---------------------------------------------------------------------------
# coefficient_of_variation
# ---------------------------------------------------------------------------

class TestCoefficientOfVariation:
    def test_raises_on_single_spike(self):
        with pytest.raises(ValueError, match="Need at least 2 spikes"):
            coefficient_of_variation(np.array([1.0]))

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="Need at least 2 spikes"):
            coefficient_of_variation(np.array([], dtype=np.float64))

    def test_poisson_approx_1(self):
        """Homogeneous Poisson → CV ≈ 1 (± 0.15 for 5000 spikes)."""
        rng = np.random.default_rng(0)
        isi = rng.exponential(50.0, size=5000)  # mean ISI = 50 ms ≈ 20 Hz
        spikes = np.cumsum(isi)
        cv = coefficient_of_variation(spikes)
        assert 0.85 <= cv <= 1.15, f"Poisson CV was {cv:.3f}"

    def test_regular_near_0(self):
        """Perfectly regular → CV ≈ 0."""
        spikes = np.arange(0, 1000, 50.0)
        cv = coefficient_of_variation(spikes)
        assert cv < 0.01, f"Regular CV was {cv:.4f}"

    def test_bursty_gt_1(self):
        """Irregular / bursting → CV > 1."""
        # Alternating very short (burst) and very long ISIs
        isis = np.concatenate([
            np.full(15, 2.0),    # tight bursts
            np.full(15, 1000.0), # long silence between bursts
        ])
        rng = np.random.default_rng(1)
        rng.shuffle(isis)
        spikes = np.cumsum(isis)
        cv = coefficient_of_variation(spikes)
        assert cv > 1.0, f"Bursty CV was {cv:.3f}"


# ---------------------------------------------------------------------------
# mean_firing_rate
# ---------------------------------------------------------------------------

class TestMeanFiringRate:
    def test_zero_spikes(self):
        assert mean_firing_rate(np.array([]), 10_000.0) == 0.0

    def test_known_rate(self):
        """100 spikes in 10 s → 10 Hz."""
        spikes = np.linspace(50, 9950, 100)
        rate = mean_firing_rate(spikes, 10_000.0)
        assert abs(rate - 10.0) < 0.1

    def test_zero_duration(self):
        assert mean_firing_rate(np.array([1.0]), 0.0) == 0.0


# ---------------------------------------------------------------------------
# isi_histogram
# ---------------------------------------------------------------------------

class TestISIHistogram:
    def test_empty(self):
        counts, edges = isi_histogram(np.array([]), bin_width_ms=5.0)
        assert len(counts) == 1
        assert counts[0] == 0

    def test_poisson_exponential_shape(self):
        """Poisson ISIs are exponentially distributed → more low-ISI counts."""
        rng = np.random.default_rng(42)
        isis = rng.exponential(50.0, size=3000)
        spikes = np.cumsum(isis)
        counts, _ = isi_histogram(spikes, bin_width_ms=5.0)
        # First bin should have the highest count (exponential decay)
        assert counts[0] > counts[-1], "ISI histogram should decay for Poisson"

    def test_regular_single_peak(self):
        """Regular spiking → narrow ISI histogram peak."""
        spikes = np.arange(0, 5000, 50.0)
        counts, edges = isi_histogram(spikes, bin_width_ms=2.0)
        # Most counts in bin near 50 ms
        peak_idx = np.argmax(counts)
        bin_center = edges[peak_idx] + np.diff(edges[peak_idx:peak_idx+2])[0] / 2
        assert abs(bin_center - 50.0) < 2.0

    def test_max_isi_cutoff(self):
        """max_isi_ms should truncate outliers."""
        spikes = np.array([0, 10, 15, 1000, 1010])
        counts, edges = isi_histogram(spikes, max_isi_ms=50.0)
        # ISI of 985 ms should be excluded
        assert counts.sum() < 4  # at least one ISI was cut off


# ---------------------------------------------------------------------------
# autocorrelogram
# ---------------------------------------------------------------------------

class TestAutocorrelogram:
    def test_zero_lag_peak(self):
        """Autocorrelogram should peak at zero lag."""
        rng = np.random.default_rng(99)
        isi = rng.exponential(50.0, size=1000)
        spikes = np.cumsum(isi)
        corr, lags = autocorrelogram(spikes, bin_width_ms=2.0)
        center = np.argmin(np.abs(lags))
        assert corr[center] >= np.max(corr) - 1e-9

    def test_regular_has_peaks(self):
        """Regular firing → autocorrelogram peaks at ISI multiples."""
        spikes = np.arange(0, 5000, 50.0)
        corr, lags = autocorrelogram(spikes, bin_width_ms=5.0)
        # Find first positive peak away from zero
        peak_indices = np.where(
            (corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:])
        )[0] + 1
        peak_lags = lags[peak_indices]
        # Should have a peak near 50 ms
        has_peak = any(abs(pl - 50.0) < 10.0 for pl in peak_lags)
        assert has_peak, f"No peak near 50 ms; peaks at {peak_lags}"

    def test_output_lengths(self):
        """Output arrays should have matching lengths."""
        spikes = np.array([0, 25, 50, 100, 150, 200])
        corr, lags = autocorrelogram(spikes)
        assert len(corr) == len(lags)


# ---------------------------------------------------------------------------
# analyze (full pipeline)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_dict_with_keys(self):
        spikes = np.cumsum(np.random.default_rng(0).exponential(50, 500))
        result = analyze(spikes, duration_ms=25_000.0, name="test")
        expected_keys = {"name", "spikes", "spike_times", "isi_mean", "isi_std",
                         "cv", "firing_rate_hz", "isi_counts",
                         "isi_edges", "corr", "lags"}
        assert expected_keys.issubset(result.keys())

    def test_known_cv(self):
        spikes = np.arange(0, 1000, 50.0)
        result = analyze(spikes, duration_ms=10_000.0)
        assert result["cv"] < 0.01

    def test_known_rate(self):
        spikes = np.linspace(50, 9950, 100)
        result = analyze(spikes, duration_ms=10_000.0)
        assert abs(result["firing_rate_hz"] - 10.0) < 0.1

    def test_single_spike(self):
        spikes = np.array([500.0])
        result = analyze(spikes, duration_ms=10_000.0)
        assert result["spikes"] == 1
        assert result["cv"] == 0.0
        assert result["isi_mean"] == 0.0
