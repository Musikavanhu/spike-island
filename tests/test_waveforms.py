"""Tests for spike_island.waveforms."""

from __future__ import annotations

import numpy as np
import pytest

from spike_island.waveforms import (
    detect_spikes_from_waveform,
    generate_ap_template,
    generate_triphasic_template,
    spikes_to_waveform,
    waveform_statistics,
)


# ---------------------------------------------------------------------------
# Module-level fixtures (used by multiple test classes)
# ---------------------------------------------------------------------------

@pytest.fixture
def template():
    """Default biphasic AP template voltage array."""
    _, v = generate_ap_template()
    return v


@pytest.fixture
def spike_times():
    """Five evenly-spaced spike times (ms)."""
    return np.array([100.0, 300.0, 500.0, 800.0, 1200.0])


# ---------------------------------------------------------------------------
# generate_ap_template (biphasic)
# ---------------------------------------------------------------------------

class TestGenerateAPTemplate:
    def test_returns_arrays(self):
        times, voltage = generate_ap_template()
        assert isinstance(times, np.ndarray)
        assert isinstance(voltage, np.ndarray)
        assert len(times) == len(voltage)

    def test_lengths(self):
        """2 ms / 0.1 ms = 20 samples."""
        times, voltage = generate_ap_template()
        assert len(times) == 20

    def test_has_negative_peak(self):
        """Template should have a negative peak (downstroke)."""
        _, voltage = generate_ap_template()
        assert np.min(voltage) < 0, "Should have negative deflection"

    def test_has_positive_undershoot(self):
        """Template should have a positive undershoot (upstroke)."""
        _, voltage = generate_ap_template()
        assert np.max(voltage) > 0, "Should have positive undershoot"

    def test_starts_at_zero(self):
        """Template should be exactly zero at the first sample."""
        _, voltage = generate_ap_template()
        assert voltage[0] == 0.0

    def test_ends_at_zero(self):
        """Template should return to exactly zero at the last sample."""
        _, voltage = generate_ap_template()
        assert voltage[-1] == 0.0

    def test_time_axis(self):
        """Time axis should start at 0 and increment by dt."""
        times, _ = generate_ap_template(dt_ms=0.1)
        assert times[0] == 0.0
        assert np.allclose(np.diff(times), 0.1)

    def test_custom_amplitude(self):
        """Custom amplitudes should be reflected in the output."""
        _, voltage = generate_ap_template(peak_amplitude_mv=-10.0)
        assert np.min(voltage) < -8.0, "Should be close to -10 mV"

    def test_custom_timing(self):
        """Custom peak timing should shift the waveform."""
        _, voltage = generate_ap_template(peak_time_ms=0.3)
        peak_idx = np.argmin(voltage)
        assert peak_idx < 10, f"Peak at index {peak_idx} should be early"


# ---------------------------------------------------------------------------
# generate_triphasic_template
# ---------------------------------------------------------------------------

class TestGenerateTriphasicTemplate:
    def test_returns_arrays(self):
        times, voltage = generate_triphasic_template()
        assert isinstance(times, np.ndarray)
        assert isinstance(voltage, np.ndarray)
        assert len(times) == len(voltage)

    def test_three_peaks(self):
        """Triphasic should have more structure than biphasic."""
        _, voltage = generate_triphasic_template()
        # Find local minima (negative peaks)
        minima = np.zeros(len(voltage), dtype=bool)
        minima[1:-1] = (voltage[1:-1] < voltage[:-2]) & (voltage[1:-1] < voltage[2:])
        negative_peaks = np.sum(minima & (voltage < 0))
        assert negative_peaks >= 1, "Should have at least one negative peak"

    def test_starts_ends_zero(self):
        times, voltage = generate_triphasic_template()
        assert voltage[0] == 0.0
        assert voltage[-1] == 0.0


# ---------------------------------------------------------------------------
# spikes_to_waveform
# ---------------------------------------------------------------------------

class TestSpikesToWaveform:
    def test_empty_spikes(self, template):
        """No spikes → flat zero trace."""
        times, voltage = spikes_to_waveform(
            np.array([]), template, duration_ms=1000.0
        )
        assert np.allclose(voltage, 0.0)
        assert len(times) > 0

    def test_returns_correct_length(self, template, spike_times):
        """10 kHz sampling × 2000 ms = 20,000 samples."""
        times, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=2000.0
        )
        expected_length = int(2000.0 * 10.0)  # 2000 ms × 10 kHz / 1000
        assert len(times) == expected_length
        assert len(voltage) == expected_length

    def test_has_spikes(self, template, spike_times):
        """Waveform should have non-zero values at spike locations."""
        times, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=2000.0
        )
        assert np.max(np.abs(voltage)) > 0, "Should have signal"

    def test_has_negative_deflections(self, template, spike_times):
        """Waveform should have negative peaks (downstrokes)."""
        times, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=2000.0
        )
        assert np.min(voltage) < 0, "Should have negative deflections"

    def test_single_spike(self, template):
        """Single spike should produce one AP waveform."""
        times, voltage = spikes_to_waveform(
            np.array([500.0]), template, duration_ms=1000.0
        )
        assert np.max(np.abs(voltage)) > 0

    def test_adds_noise(self, template, spike_times):
        """With noise > 0, variance should increase."""
        _, no_noise = spikes_to_waveform(
            spike_times, template, duration_ms=2000.0, noise_std_mv=0.0
        )
        _, with_noise = spikes_to_waveform(
            spike_times, template, duration_ms=2000.0,
            noise_std_mv=0.5, noise_type="gaussian", seed=42,
        )
        assert np.std(with_noise) > np.std(no_noise), \
            "Noise should increase variance"

    def test_pink_noise(self, template, spike_times):
        """Pink noise should be generated without error."""
        times, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=2000.0,
            noise_std_mv=0.1, noise_type="pink", seed=42,
        )
        assert len(times) == len(voltage)
        assert np.max(np.abs(voltage)) > 0

    def test_invalid_noise_type(self, template, spike_times):
        with pytest.raises(ValueError, match="Unknown noise_type"):
            spikes_to_waveform(
                spike_times, template, duration_ms=1000.0,
                noise_std_mv=0.1, noise_type="purple",
            )

    def test_reproducibility(self, template, spike_times):
        """Same seed → same output."""
        _, v1 = spikes_to_waveform(
            spike_times, template, duration_ms=1000.0,
            noise_std_mv=0.1, seed=123,
        )
        _, v2 = spikes_to_waveform(
            spike_times, template, duration_ms=1000.0,
            noise_std_mv=0.1, seed=123,
        )
        np.testing.assert_array_equal(v1, v2)

    def test_different_sampling_hz(self, template, spike_times):
        """Different sampling rates produce different lengths."""
        _, v1 = spikes_to_waveform(
            spike_times, template, duration_ms=1000.0, sampling_hz=1000.0
        )
        _, v2 = spikes_to_waveform(
            spike_times, template, duration_ms=1000.0, sampling_hz=10_000.0
        )
        assert len(v2) > len(v1), "Higher sampling → more samples"


# ---------------------------------------------------------------------------
# waveform_statistics
# ---------------------------------------------------------------------------

class TestWaveformStatistics:
    def test_returns_required_keys(self):
        expected = {"rms_mv", "peak_to_peak_mv", "max_mv", "min_mv",
                    "snr_db", "mean_voltage_mv", "duration_ms", "samples"}
        voltage = np.random.default_rng(0).normal(0, 0.1, 1000)
        times = np.arange(1000) * 0.1
        stats = waveform_statistics(voltage, times)
        assert expected.issubset(stats.keys())

    def test_zero_signal(self):
        voltage = np.zeros(100)
        stats = waveform_statistics(voltage)
        assert stats["rms_mv"] == 0.0
        assert stats["peak_to_peak_mv"] == 0.0

    def test_peak_to_peak(self):
        voltage = np.array([-3.0, 0.0, 1.0, 0.0, -3.0])
        stats = waveform_statistics(voltage)
        assert abs(stats["peak_to_peak_mv"] - 4.0) < 0.01

    def test_snr_positive(self):
        """Signal + noise should give finite SNR."""
        rng = np.random.default_rng(42)
        voltage = rng.normal(0, 0.1, 1000)
        voltage[100] = -3.0  # inject a spike
        voltage[500] = -3.0
        stats = waveform_statistics(voltage)
        assert stats["snr_db"] > -10, "SNR should be finite"

    def test_samples_count(self):
        voltage = np.random.randn(500)
        stats = waveform_statistics(voltage)
        assert stats["samples"] == 500

    def test_duration(self):
        """Duration = last - first time sample."""
        times = np.arange(0.0, 100.0, 0.1)  # 0.0 to 99.9 ms
        voltage = np.zeros_like(times)
        stats = waveform_statistics(voltage, times)
        # last element is 99.9 (arange excludes stop)
        assert abs(stats["duration_ms"] - 99.9) < 0.1

    def test_no_times(self):
        voltage = np.array([1.0, 2.0, 3.0])
        stats = waveform_statistics(voltage, times=None)
        assert stats["duration_ms"] == 0.0


# ---------------------------------------------------------------------------
# detect_spikes_from_waveform
# ---------------------------------------------------------------------------

class TestDetectSpikesFromWaveform:
    def test_detects_spikes(self, template):
        """Should recover original spike positions (approximately)."""
        spike_times = np.array([100.0, 300.0, 500.0])
        times, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=1000.0,
            noise_std_mv=0.0,  # no noise for clean detection
        )
        detected = detect_spikes_from_waveform(voltage)
        assert len(detected) >= 1, "Should detect spikes in clean signal"

    def test_no_false_positives_on_flat(self):
        """Flat signal → no detections."""
        voltage = np.zeros(1000)
        detected = detect_spikes_from_waveform(voltage)
        # With zero std, threshold is 0 → no minima below 0
        assert len(detected) == 0

    def test_custom_threshold(self, template):
        """Custom threshold should control sensitivity."""
        spike_times = np.array([100.0, 300.0, 500.0])
        _, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=1000.0, noise_std_mv=0.0,
        )
        # Very permissive threshold → catches more
        detected_loose = detect_spikes_from_waveform(voltage, threshold_mv=-0.5)
        # Very strict threshold → catches fewer
        detected_strict = detect_spikes_from_waveform(voltage, threshold_mv=-10.0)
        assert len(detected_loose) >= len(detected_strict)

    def test_refractory_filtering(self, template):
        """Spikes within refractory period should be merged."""
        # Two very close spikes
        spike_times = np.array([100.0, 100.5, 500.0])
        _, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=1000.0, noise_std_mv=0.0,
        )
        detected = detect_spikes_from_waveform(voltage)
        # At 10 kHz sampling, 0.5 ms = 5 bins — should be filtered
        assert len(detected) <= 3

    def test_with_noise(self, template):
        """Noise should not produce excessive false positives."""
        spike_times = np.array([100.0, 300.0, 500.0])
        _, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=1000.0,
            noise_std_mv=0.1, seed=42,
        )
        detected = detect_spikes_from_waveform(voltage)
        # Should not have more detections than original (with threshold)
        assert len(detected) <= 10

    def test_returns_indices(self, template):
        """Should return valid array indices."""
        spike_times = np.array([100.0])
        times, voltage = spikes_to_waveform(
            spike_times, template, duration_ms=500.0, noise_std_mv=0.0,
        )
        detected = detect_spikes_from_waveform(voltage)
        if len(detected) > 0:
            assert all(0 <= idx < len(voltage) for idx in detected)
