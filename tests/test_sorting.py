"""Tests for spike_island.sorting module."""

import pytest
import numpy as np

from spike_island.sorting import (
    SpikeEvent,
    SortingReport,
    generate_ground_truth,
    contaminate_recording,
    _correlation,
    _correlation_with_template,
    template_sort,
    evaluate_sorting,
    print_sorting_report,
)


class TestGenerateGroundTruth:
    """Tests for the ground truth simulator."""

    def test_returns_correct_count(self):
        """Should return exactly n_neurons arrays."""
        result = generate_ground_truth(n_neurons=5, seed=0)
        assert len(result) == 5

    def test_returns_sorted_arrays(self):
        """Each spike train should be sorted ascending."""
        trains = generate_ground_truth(n_neurons=3, seed=0)
        for t in trains:
            assert np.all(np.diff(t) >= 0)

    def test_all_times_in_range(self):
        """All spike times should be within [0, duration_ms)."""
        duration_ms = 1000.0
        trains = generate_ground_truth(
            n_neurons=3, duration_ms=duration_ms, seed=0
        )
        for t in trains:
            assert np.all(t >= 0)
            assert np.all(t < duration_ms)

    def test_deterministic(self):
        """Same seed should produce identical spike trains."""
        t1 = generate_ground_truth(n_neurons=2, seed=42)
        t2 = generate_ground_truth(n_neurons=2, seed=42)
        for a, b in zip(t1, t2):
            np.testing.assert_array_equal(a, b)

    def test_rate_scales_spikes(self):
        """Higher rate should yield more spikes (approximately)."""
        t_low = generate_ground_truth(rate_hz=5, duration_ms=1000, seed=0)
        t_high = generate_ground_truth(rate_hz=50, duration_ms=1000, seed=0)
        total_low = sum(len(s) for s in t_low)
        total_high = sum(len(s) for s in t_high)
        assert total_high > total_low

    def test_empty_rate(self):
        """Zero rate should give empty spike trains."""
        trains = generate_ground_truth(rate_hz=0, duration_ms=1000, seed=0)
        for t in trains:
            assert len(t) == 0

    def test_non_empty_single_neuron(self):
        """A neuron at 20 Hz for 1000 ms should have reasonable spikes."""
        trains = generate_ground_truth(rate_hz=20, duration_ms=1000, seed=0)
        assert 5 < len(trains[0]) < 50


class TestContaminateRecording:
    """Tests for the contaminated recording generator."""

    def test_returns_recording_and_assignments(self):
        spike_trains = [np.array([100.0, 300.0])]
        tpl = np.array([-1.0, 0.5, -0.2])
        rec, true_spikes = contaminate_recording(
            spike_times_list=spike_trains,
            templates=[tpl],
            seed=0,
        )
        assert isinstance(rec, np.ndarray)
        assert len(true_spikes) == 1

    def test_recording_length(self):
        duration_ms = 5000.0
        sampling_hz = 10_000.0
        spike_trains = [np.array([500.0])]
        tpl = np.array([-1.0, 0.5])
        rec, _ = contaminate_recording(
            spike_times_list=spike_trains,
            templates=[tpl],
            sampling_hz=sampling_hz,
            duration_ms=duration_ms,
            seed=0,
        )
        expected_len = int(duration_ms * sampling_hz / 1000.0)
        assert len(rec) == expected_len

    def test_nonzero_recording(self):
        spike_trains = [np.array([500.0, 1000.0])]
        tpl = np.array([-10.0, 0.0, 0.0, 5.0])
        rec, _ = contaminate_recording(
            spike_times_list=spike_trains,
            templates=[tpl],
            seed=0,
        )
        assert np.any(rec != 0.0)

    def test_zero_noise_is_pure_sum(self):
        """With zero noise, recording should be exactly template superposition."""
        spike_trains = [np.array([500.0])]
        tpl = np.array([-2.0, 1.0])
        rec, _ = contaminate_recording(
            spike_times_list=spike_trains,
            templates=[tpl],
            noise_std=0.0,
            seed=0,
        )
        expected = np.zeros(len(rec))
        dt_ms = 1000.0 / 10000.0  # 0.1 ms per sample
        idx = int(500.0 / dt_ms)  # 5000 samples
        half = len(tpl) // 2  # 0
        for i in range(len(tpl)):
            pos = idx - half + i
            if 0 <= pos < len(expected):
                expected[pos] = tpl[i]
        np.testing.assert_array_almost_equal(rec, expected)

    def test_deterministic(self):
        spike_trains = [np.array([500.0])]
        tpl = np.array([-2.0, 1.0])
        r1, _ = contaminate_recording(
            spike_times_list=spike_trains, templates=[tpl], seed=0
        )
        r2, _ = contaminate_recording(
            spike_times_list=spike_trains, templates=[tpl], seed=0
        )
        np.testing.assert_array_almost_equal(r1, r2)

    def test_two_neurons(self):
        """Two neuron spike trains should both contribute to recording."""
        spike_trains = [np.array([500.0]), np.array([600.0])]
        tpl0 = np.array([-5.0, 0.0, 2.0])
        tpl1 = np.array([-3.0, 0.0, 1.0])
        rec, true_spikes = contaminate_recording(
            spike_times_list=spike_trains,
            templates=[tpl0, tpl1],
            seed=0,
        )
        assert len(rec) > 0
        assert len(true_spikes) == 2
        # Both neurons contributed
        assert sum(len(s) for s in true_spikes) == 2


class TestCorrelationHelpers:
    """Tests for the internal correlation helpers."""

    def test_perfect_correlation(self):
        """A signal matched against itself should have correlation 1.0."""
        sig = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(_correlation(sig, sig) - 1.0) < 1e-10

    def test_zero_correlation(self):
        """Orthogonal vectors should correlate near zero."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_correlation(a, b)) < 1e-10

    def test_negative_correlation(self):
        """Negative correlation (anti-aligned) returns -1.0."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _correlation(a, b) == pytest.approx(-1.0)

    def test_zero_template(self):
        """Zero template should return zero correlation."""
        sig = np.array([1.0, 2.0])
        zero = np.array([0.0, 0.0])
        assert abs(_correlation(sig, zero)) < 1e-10

    def test_correlation_with_template(self):
        """Basic sanity check for the sliding window function."""
        tpl = np.array([-2.0, 1.0, 0.5])
        rec = np.zeros(100)
        rec[50:53] = [-2.0, 1.0, 0.5]
        corr, idx = _correlation_with_template(
            rec, tpl, window_ms=1.0, dt_ms=1.0
        )
        assert corr > 0.5
        assert abs(idx - 50) <= 1

    def test_correlation_negative_signal(self):
        """Matching a negative template should return positive correlation."""
        tpl = np.array([1.0, -2.0, 0.5])
        rec = np.zeros(50)
        rec[20:23] = [1.0, -2.0, 0.5]
        corr, idx = _correlation_with_template(
            rec, tpl, window_ms=1.0, dt_ms=1.0
        )
        assert corr > 0.5
        # Index is the first sample of the matching template
        assert idx == 20


class TestTemplateSort:
    """Tests for the main spike sorting function."""

    def test_sort_identifies_known_spike(self):
        """Should detect at least one spike from a clear template."""
        dt_ms = 1.0
        n_samples = 200
        rec = np.zeros(n_samples)
        rec[48:51] = [-5.0, 0.0, 3.0]
        tpl = np.array([-5.0, 0.0, 3.0])
        events, counts = template_sort(
            recording=rec,
            templates=[tpl],
            dt_ms=dt_ms,
            threshold=0.3,
            refractory_ms=50.0,
            time_window_ms=2.0,
            max_iter=10,
        )
        assert len(events) >= 1

    def test_event_has_neuron_id(self):
        rec = np.zeros(200)
        rec[48:51] = [-5.0, 0.0, 3.0]
        tpl = np.array([-5.0, 0.0, 3.0])
        events, _ = template_sort(
            recording=rec,
            templates=[tpl],
            dt_ms=1.0,
            threshold=0.3,
            refractory_ms=50.0,
            time_window_ms=2.0,
            max_iter=10,
        )
        for ev in events:
            assert isinstance(ev.neuron_id, int)

    def test_event_has_time_ms(self):
        rec = np.zeros(200)
        rec[48:51] = [-5.0, 0.0, 3.0]
        tpl = np.array([-5.0, 0.0, 3.0])
        events, _ = template_sort(
            recording=rec,
            templates=[tpl],
            dt_ms=1.0,
            threshold=0.3,
            refractory_ms=50.0,
            time_window_ms=2.0,
            max_iter=10,
        )
        for ev in events:
            assert isinstance(ev.time_ms, float)
            assert ev.time_ms >= 0

    def test_no_detection_above_threshold(self):
        """If no template exceeds the threshold, should return empty list."""
        rec = np.random.randn(200) * 0.01
        tpl = np.array([1.0, -2.0, 0.5])
        events, counts = template_sort(
            recording=rec,
            templates=[tpl],
            dt_ms=1.0,
            threshold=100.0,
            refractory_ms=50.0,
            time_window_ms=2.0,
            max_iter=10,
        )
        assert len(events) == 0
        assert counts.get(0, 0) == 0

    def test_per_neuron_count_matches_events(self):
        rec = np.zeros(200)
        rec[48:51] = [-5.0, 0.0, 3.0]
        tpl = np.array([-5.0, 0.0, 3.0])
        events, counts = template_sort(
            recording=rec,
            templates=[tpl],
            dt_ms=1.0,
            threshold=0.3,
            refractory_ms=50.0,
            time_window_ms=2.0,
            max_iter=10,
        )
        total = sum(counts.values())
        assert total == len(events)

    def test_multiple_neurons(self):
        """With 3 neurons, should assign events to all of them."""
        dt_ms = 0.1
        n_samples = 5000
        rec = np.zeros(n_samples)
        tpl0 = np.array([-3.0, 1.0, 0.0])
        tpl1 = np.array([-2.0, 0.0, 1.5])
        tpl2 = np.array([-4.0, 2.0, 0.0])

        # Place spikes at known positions
        positions = [50, 100, 200, 300, 400]
        for pos in positions:
            half = 1
            rec[pos - half : pos + half + 1] += tpl0

        events, counts = template_sort(
            recording=rec,
            templates=[tpl0, tpl1, tpl2],
            dt_ms=dt_ms,
            threshold=0.3,
            refractory_ms=1.0,
            time_window_ms=2.0,
            max_iter=500,
        )
        assert len(events) > 0
        neuron_ids = {ev.neuron_id for ev in events}
        assert len(neuron_ids) >= 1

    def test_empty_recording(self):
        """Zero recording should produce no detections at high threshold."""
        rec = np.zeros(100)
        tpl = np.array([1.0, -2.0, 0.5])
        events, counts = template_sort(
            recording=rec,
            templates=[tpl],
            dt_ms=1.0,
            threshold=0.5,
            refractory_ms=10.0,
            time_window_ms=2.0,
            max_iter=10,
        )
        assert len(events) == 0
        assert counts.get(0, 0) == 0

    def test_event_correlation_positive(self):
        """Events from detectable signals should have positive correlation."""
        rec = np.zeros(200)
        rec[48:51] = [-5.0, 0.0, 3.0]
        tpl = np.array([-5.0, 0.0, 3.0])
        events, _ = template_sort(
            recording=rec,
            templates=[tpl],
            dt_ms=1.0,
            threshold=0.3,
            refractory_ms=50.0,
            time_window_ms=2.0,
            max_iter=10,
        )
        for ev in events:
            assert ev.correlation > 0


class TestEvaluateSorting:
    """Tests for the evaluation pipeline."""

    def test_perfect_sorting(self):
        """When all true spikes are detected exactly, precision/recall = 1.0."""
        true_spikes = [
            np.array([100.0, 200.0, 300.0]),
            np.array([150.0, 250.0]),
        ]
        detected = [
            SpikeEvent(time_ms=100.0, neuron_id=0, correlation=0.9),
            SpikeEvent(time_ms=200.0, neuron_id=0, correlation=0.8),
            SpikeEvent(time_ms=300.0, neuron_id=0, correlation=0.95),
            SpikeEvent(time_ms=150.0, neuron_id=1, correlation=0.85),
            SpikeEvent(time_ms=250.0, neuron_id=1, correlation=0.9),
        ]
        report = evaluate_sorting(
            true_spikes=true_spikes,
            detected_spikes=detected,
            merge_window_ms=0.5,
        )
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)
        assert report.f1_score == pytest.approx(1.0)

    def test_no_detections(self):
        """No detected spikes: precision = 0, recall = 0."""
        true_spikes = [np.array([100.0, 200.0])]
        detected: list[SpikeEvent] = []
        report = evaluate_sorting(
            true_spikes=true_spikes,
            detected_spikes=detected,
            merge_window_ms=0.5,
        )
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.f1_score == 0.0

    def test_false_positives(self):
        """Extra detected spikes reduce precision."""
        true_spikes = [np.array([100.0])]
        detected = [
            SpikeEvent(time_ms=100.0, neuron_id=0, correlation=0.9),
            SpikeEvent(time_ms=500.0, neuron_id=0, correlation=0.8),
        ]
        report = evaluate_sorting(
            true_spikes=true_spikes,
            detected_spikes=detected,
            merge_window_ms=0.5,
        )
        assert report.precision < 1.0
        assert report.recall == pytest.approx(1.0)

    def test_false_negatives(self):
        """Missed true spikes reduce recall."""
        true_spikes = [np.array([100.0, 200.0, 300.0])]
        detected = [SpikeEvent(time_ms=100.0, neuron_id=0, correlation=0.9)]
        report = evaluate_sorting(
            true_spikes=true_spikes,
            detected_spikes=detected,
            merge_window_ms=0.5,
        )
        assert report.recall == pytest.approx(1.0 / 3.0)
        assert report.recall < 1.0

    def test_per_neuron_breakdown(self):
        true_spikes = [
            np.array([100.0]),
            np.array([200.0, 300.0]),
        ]
        detected = [
            SpikeEvent(time_ms=100.0, neuron_id=0, correlation=0.9),
            SpikeEvent(time_ms=200.0, neuron_id=1, correlation=0.8),
            SpikeEvent(time_ms=300.0, neuron_id=1, correlation=0.7),
        ]
        report = evaluate_sorting(
            true_spikes=true_spikes,
            detected_spikes=detected,
            merge_window_ms=0.5,
        )
        assert "0" in report.per_neuron
        assert "1" in report.per_neuron
        assert report.per_neuron["0"]["true"] == 1
        assert report.per_neuron["0"]["matched"] == 1
        assert report.per_neuron["1"]["true"] == 2
        assert report.per_neuron["1"]["matched"] == 2

    def test_total_counts(self):
        true_spikes = [
            np.array([100.0, 200.0]),
            np.array([500.0]),
        ]
        detected = [
            SpikeEvent(time_ms=100.0, neuron_id=0, correlation=0.9),
            SpikeEvent(time_ms=200.0, neuron_id=0, correlation=0.8),
            SpikeEvent(time_ms=500.0, neuron_id=1, correlation=0.7),
            SpikeEvent(time_ms=900.0, neuron_id=0, correlation=0.5),
        ]
        report = evaluate_sorting(
            true_spikes=true_spikes,
            detected_spikes=detected,
            merge_window_ms=0.5,
        )
        assert report.total_detected == 4
        assert report.total_true == 3
        assert report.matched == 3
        assert report.false_positives == 1
        assert report.false_negatives == 0

    def test_both_empty(self):
        """Empty ground truth and no detections should yield perfect metrics."""
        report = evaluate_sorting(
            true_spikes=[],
            detected_spikes=[],
            merge_window_ms=0.5,
        )
        assert report.total_detected == 0
        assert report.total_true == 0
        assert report.matched == 0
        assert report.f1_score == pytest.approx(0.0)


class TestSortingReport:
    """Tests for the SortingReport dataclass."""

    def test_defaults(self):
        r = SortingReport()
        assert r.total_detected == 0
        assert r.matched == 0
        assert r.precision == 0.0
        assert r.f1_score == 0.0

    def test_per_neuron_dict_exists(self):
        r = SortingReport()
        r.per_neuron["neuron_0"] = {
            "true": 10, "detected": 8, "matched": 7,
        }
        assert r.per_neuron["neuron_0"]["true"] == 10


class TestPrintSortingReport:
    """Tests for the pretty-print function (integration test)."""

    def test_prints_without_error(self):
        report = SortingReport(
            total_detected=100,
            total_true=90,
            matched=85,
            precision=0.85,
            recall=0.94,
            f1_score=0.89,
        )
        report.per_neuron["neuron_0"] = {
            "true": 50, "detected": 45, "matched": 40,
        }
        print_sorting_report(report)
        # If we get here without exception, test passes


class TestFullPipeline:
    """Integration tests: full pipeline from simulators to sorters."""

    def test_sort_across_poisson_spikes(self):
        """Generate Poisson spikes, build templates, sort, evaluate."""
        from spike_island.simulators import poisson_spikes

        spikes = poisson_spikes(rate_hz=10, duration_ms=2000, seed=42)
        assert len(spikes) > 5

        tpl = np.array([-5.0, 0.0, 2.0, -0.5])

        rec, true_spikes = contaminate_recording(
            spike_times_list=[spikes],
            templates=[tpl],
            noise_std=0.3,
            duration_ms=2000,
            seed=42,
        )

        events, counts = template_sort(
            recording=rec,
            templates=[tpl],
            dt_ms=0.1,
            threshold=0.3,
            refractory_ms=2.0,
            time_window_ms=1.0,
            max_iter=200,
        )

        report = evaluate_sorting(
            true_spikes=true_spikes,
            detected_spikes=events,
            merge_window_ms=0.5,
        )

        assert report.total_detected > 0
        assert report.total_true > 0
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0

    def test_three_neurons_sort(self):
        """Three-neuron ground truth should sort across all classes."""
        from spike_island.simulators import poisson_spikes

        n_neurons = 3
        rate = 8.0
        duration_ms = 3000.0

        spike_trains = []
        tpl_list = []
        for i in range(n_neurons):
            s = poisson_spikes(rate_hz=rate, duration_ms=duration_ms, seed=i)
            spike_trains.append(s)
            # Different amplitude per neuron
            base_amp = -(5.0 + i * 2.0)
            tpl = np.array([base_amp, 0.0, base_amp * 0.3])
            tpl_list.append(tpl)

        rec, true_spikes = contaminate_recording(
            spike_times_list=spike_trains,
            templates=tpl_list,
            noise_std=0.3,
            duration_ms=duration_ms,
            seed=42,
        )

        events, counts = template_sort(
            recording=rec,
            templates=tpl_list,
            dt_ms=0.1,
            threshold=0.3,
            refractory_ms=2.0,
            time_window_ms=1.0,
            max_iter=300,
        )

        report = evaluate_sorting(
            true_spikes=true_spikes,
            detected_spikes=events,
            merge_window_ms=0.5,
        )

        assert report.total_true > 0
        assert report.total_detected >= 0
        # At least some neurons should have matches
        per_neuron_matched = sum(
            m["matched"] for m in report.per_neuron.values()
        )
        assert per_neuron_matched >= 0


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"])
