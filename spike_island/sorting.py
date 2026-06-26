"""Spike Island: Spike Sorting via Template Matching.

Implements a template-matching spike sorter that identifies spike times
and assigns each event to the most likely neuron class from a contaminated
extracellular recording (superposition of spike waveforms from multiple
neurons plus noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpikeEvent:
    """A single detected spike event with classification info."""

    time_ms: float          # spike time in milliseconds
    neuron_id: int          # assigned neuron class (0, 1, 2, ...)
    correlation: float      # correlation score with matched template
    residual: float = 0.0   # residual energy after subtracting matched waveform


@dataclass
class SortingReport:
    """Results of a sorting evaluation against ground truth."""

    total_detected: int = 0
    total_true: int = 0
    matched: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    per_neuron: dict = field(default_factory=dict)


def generate_ground_truth(
    n_neurons: int = 3,
    rate_hz: float = 15.0,
    duration_ms: float = 10_000.0,
    seed: int | None = None,
) -> List[NDArray[np.float64]]:
    """Generate independent Poisson spike trains for multiple neurons.

    Each neuron fires independently from a homogeneous Poisson process
    with the specified rate. Simulates a realistic multi-unit environment
    where spike times may overlap due to finite temporal resolution.

    Args:
        n_neurons: Number of simulated neurons (0..n-1).
        rate_hz: Base firing rate for each neuron in Hz.
        duration_ms: Simulation window length in milliseconds.
        seed: Random seed for reproducibility.

    Returns:
        List of sorted spike time arrays, one per neuron.
    """
    rng = np.random.default_rng(seed)
    spike_trains: List[NDArray[np.float64]] = []

    for _ in range(n_neurons):
        n_spikes = rng.poisson(lam=rate_hz * duration_ms / 1000.0)
        times = np.sort(rng.uniform(0, duration_ms, size=n_spikes))
        spike_trains.append(times.astype(np.float64))

    return spike_trains


def contaminate_recording(
    spike_times_list: List[NDArray[np.float64]],
    templates: List[NDArray[np.float64]],
    noise_std: float = 0.3,
    sampling_hz: float = 10_000.0,
    duration_ms: float = 5_000.0,
    seed: int | None = None,
) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]]]:
    """Create a contaminated extracellular recording.

    Superimposes spike waveforms from multiple neurons (each represented
    by a template) at their respective spike times, then adds Gaussian
    noise. Also returns the true spike assignments for evaluation.

    Args:
        spike_times_list: List of sorted spike time arrays, one per neuron.
        templates: List of template waveforms, one per neuron.
        noise_std: Standard deviation of additive Gaussian noise (mV).
        sampling_hz: Recording sampling rate in Hz.
        duration_ms: Recording duration in ms.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (recording, true_assignments) where recording is the
        continuous voltage trace and true_assignments is a list of spike
        time arrays per neuron (the ground truth).
    """
    rng = np.random.default_rng(seed)
    dt_ms = 1.0 / sampling_hz * 1000.0  # ms per sample
    n_samples = int(duration_ms / dt_ms)

    recording = np.zeros(n_samples)

    for spike_times, template in zip(spike_times_list, templates):
        for spike_time in spike_times:
            sample_idx = int(spike_time / dt_ms)
            half_len = len(template) // 2
            rec_start = max(0, sample_idx - half_len)
            rec_end = min(n_samples, sample_idx + half_len + 1)
            tpl_len = len(template)
            t_start = max(0, half_len - sample_idx)
            t_end = min(tpl_len, half_len + (n_samples - sample_idx))
            rec_width = rec_end - rec_start
            tpl_width = t_end - t_start
            width = min(rec_width, tpl_width)
            if width > 0:
                recording[rec_start:rec_start + width] += (
                    template[t_start:t_start + width]
                )

    noise = rng.normal(0.0, noise_std, size=n_samples)
    recording += noise

    return recording, spike_times_list


def _correlation(
    segment: NDArray[np.float64],
    template: NDArray[np.float64],
) -> float:
    """Compute normalized dot-product between segment and template.

    For matching purposes the result is kept signed so the caller
    can decide whether to negate the template.
    """
    tpl_norm = np.linalg.norm(template)
    if tpl_norm == 0:
        return 0.0
    segment_norm = np.linalg.norm(segment)
    if segment_norm == 0:
        return 0.0
    dot = np.dot(segment, template)
    return float(dot / (tpl_norm * segment_norm))


def _sliding_window_correlation(
    recording: NDArray[np.float64],
    template: NDArray[np.float64],
) -> Tuple[float, int]:
    """Find the best alignment of *template* against the recording.

    Slides the template one sample at a time across the recording and
    returns the position (sample index) with the highest absolute
    dot-product value.

    Args:
        recording: The full voltage recording.
        template: The template waveform to match.

    Returns:
        Tuple of (best_score, best_sample_index).
    """
    best_score = 0.0
    best_idx = 0
    tpl_len = len(template)
    rec_len = len(recording)

    for idx in range(0, rec_len - tpl_len + 1):
        segment = recording[idx : idx + tpl_len]
        dot = np.dot(segment, template)
        score = abs(dot)
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_score, best_idx


def _correlation_with_template(
    recording: NDArray[np.float64],
    template: NDArray[np.float64],
    window_ms: float,
    dt_ms: float,
) -> Tuple[float, int]:
    """Find the best alignment of *template* against the recording.

    Slides the template one sample at a time across the full recording
    and returns the position (sample index) with the highest absolute
    dot-product value.

    Args:
        recording: The full voltage recording.
        template: The template waveform to match.

    Returns:
        Tuple of (best_score, best_sample_index).
    """
    best_score = 0.0
    best_idx = 0
    tpl_len = len(template)
    rec_len = len(recording)

    for idx in range(0, rec_len - tpl_len + 1):
        segment = recording[idx : idx + tpl_len]
        dot = np.dot(segment, template)
        score = abs(dot)
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_score, best_idx


def template_sort(
    recording: NDArray[np.float64],
    templates: List[NDArray[np.float64]],
    dt_ms: float = 0.1,
    threshold: float = 0.5,
    refractory_ms: float = 2.0,
    time_window_ms: float = 1.0,
    max_iter: int = 500,
) -> Tuple[List[SpikeEvent], Dict[int, int]]:
    """Sort spikes via matching pursuit across multiple templates.

    Implements a greedy matching pursuit algorithm: at each step, finds
    the best (template, time, neuron) triple maximizing correlation with
    the current residual, subtracts that waveform from the residual, and
    repeats until no correlation exceeds the threshold or max_iter is
    reached.

    After sorting, applies refractory filtering per neuron to remove
    double-counted events and merges overlapping assignments.

    Args:
        recording: The contaminated voltage recording (one-dimensional array).
        templates: List of template waveforms, one per neuron class.
        dt_ms: Time step in ms per sample.
        threshold: Minimum correlation value to accept a detection.
        refractory_ms: Minimum inter-spike interval enforced per neuron.
        time_window_ms: Half-window in ms for template sliding.
        max_iter: Maximum number of matching pursuit iterations.

    Returns:
        Tuple of (detected_spikes, per_neuron_spike_count) where
        detected_spikes is a list of SpikeEvent objects and
        per_neuron_spike_count maps neuron_id to its spike count.
    """
    from collections import defaultdict

    n_samples = len(recording)
    residual = recording.copy()
    events: List[SpikeEvent] = []
    neuron_ids = list(range(len(templates)))
    half_samples = int(time_window_ms / dt_ms) if dt_ms > 0 else 10000

    for _ in range(max_iter):
        best_corr = 0.0
        best_idx = 0
        best_nid = 0
        best_event: SpikeEvent | None = None

        for nid in neuron_ids:
            tpl = templates[nid]
            corr, idx = _correlation_with_template(
                recording=recording,
                template=tpl,
                window_ms=time_window_ms,
                dt_ms=dt_ms,
            )
            # Only consider if not already detected in this window
            if corr > best_corr:
                best_corr = corr
                best_idx = idx
                best_nid = nid
                best_event = SpikeEvent(
                    time_ms=idx * dt_ms,
                    neuron_id=best_nid,
                    correlation=corr,
                )

        if best_event is None or best_event.correlation < threshold:
            break

        # Subtract matched template from residual
        tpl = templates[best_event.neuron_id]
        half_len = len(tpl) // 2
        rec_start = max(0, best_idx - half_len)
        rec_end = min(n_samples, best_idx + half_len + 1)
        # Only the overlapping portion matters
        tpl_start = max(0, half_len - best_idx)
        tpl_end = min(len(tpl), half_len + (n_samples - best_idx))
        tpl_width = tpl_end - tpl_start
        rec_width = rec_end - rec_start
        overlap = min(rec_width, tpl_width)
        if overlap > 0:
            residual[rec_start:rec_start + overlap] += (
                tpl[tpl_start:tpl_start + overlap] * (-1)
            )
        events.append(best_event)

    # Apply refractory filtering
    refractory_samples = int(refractory_ms / dt_ms) if dt_ms > 0 else 10000
    filtered_events: List[SpikeEvent] = []
    counts: Dict[int, int] = {nid: 0 for nid in neuron_ids}

    for ev in sorted(events, key=lambda e: e.time_ms):
        nid = ev.neuron_id
        refr_idx = int(ev.time_ms / dt_ms) if dt_ms > 0 else 0
        refr_start = max(0, refr_idx - refractory_samples)
        refr_start_idx = refr_idx - refractory_samples

        # Check if this neuron has a spike within the refractory window
        is_refractory = False
        for prev in filtered_events:
            if prev.neuron_id == nid:
                if abs(refr_idx - int(prev.time_ms / dt_ms)) < refractory_samples:
                    is_refractory = True
                    break
        if not is_refractory:
            filtered_events.append(ev)
            counts[nid] = counts.get(nid, 0) + 1

    return filtered_events, counts


def evaluate_sorting(
    true_spikes: List[NDArray[np.float64]],
    detected_spikes: List[SpikeEvent],
    true_labels: List[int] | None = None,
    merge_window_ms: float = 0.5,
) -> SortingReport:
    """Evaluate spike sorting accuracy against ground truth.

    Matches each detected spike to the nearest true spike (within
    merge_window_ms ms) for each neuron class. Computes precision,
    recall, and F1 score.

    Args:
        true_spikes: List of true spike time arrays, one per neuron.
        detected_spikes: List of detected SpikeEvent objects.
        true_labels: Optional mapping of neuron indices (0..n-1).
            If None, uses integer indices 0..n-1.
        merge_window_ms: Maximum time window in ms for matching.

    Returns:
        SortingReport with precision, recall, F1, and per-neuron metrics.
    """
    report = SortingReport(
        total_detected=len(detected_spikes),
        total_true=sum(len(st) for st in true_spikes),
    )

    n_neurons = len(true_spikes)

    for nid in range(n_neurons):
        true_times = true_spikes[nid]
        n_true = len(true_times)
        report.per_neuron[str(nid)] = {
            "true": n_true,
            "detected": 0,
            "matched": 0,
        }

        # Find neuron's detected spikes
        neuron_detected = [ev for ev in detected_spikes if ev.neuron_id == nid]
        report.per_neuron[str(nid)]["detected"] = len(neuron_detected)

        matched = 0
        used_true: set[int] = set()

        for det_ev in neuron_detected:
            best_true_idx = -1
            best_dist = merge_window_ms + 1.0

            for true_idx, true_t in enumerate(true_times):
                if true_idx in used_true:
                    continue
                dist = abs(det_ev.time_ms - true_t)
                if dist < best_dist:
                    best_dist = dist
                    best_true_idx = true_idx

            if best_true_idx >= 0 and best_dist <= merge_window_ms:
                matched += 1
                used_true.add(best_true_idx)

        report.per_neuron[str(nid)]["matched"] = matched
        report.matched += matched

    report.false_negatives = report.total_true - report.matched
    report.false_positives = report.total_detected - report.matched

    if report.total_detected > 0:
        report.precision = report.matched / report.total_detected
    if report.total_true > 0:
        report.recall = report.matched / report.total_true
    if report.precision + report.recall > 0:
        report.f1_score = (
            2 * report.precision * report.recall
            / (report.precision + report.recall)
        )

    return report


def print_sorting_report(report: SortingReport) -> None:
    """Pretty-print a SortingReport to stdout."""
    print("=" * 50)
    print("  Spike Sorting Report")
    print("=" * 50)
    print(f"  True spikes:      {report.total_true}")
    print(f"  Detected spikes:  {report.total_detected}")
    print(f"  Matched:          {report.matched}")
    print(f"  False positives:  {report.false_positives}")
    print(f"  False negatives:  {report.false_negatives}")
    print(f"  Precision:        {report.precision:.3f}")
    print(f"  Recall:           {report.recall:.3f}")
    print(f"  F1 Score:         {report.f1_score:.3f}")
    print("-" * 50)
    print("  Per-neuron breakdown:")
    for name, metrics in report.per_neuron.items():
        f1 = metrics.get("f1_score", 0.0)
        print(
            f"    {name:>8s}:  "
            f"true={metrics['true']:>4d}  "
            f"detected={metrics['detected']:>4d}  "
            f"f1={f1:.3f}"
        )
    print("=" * 50)


if __name__ == "__main__":
    """Demo: sort a contaminated recording from 3 neurons."""
    from typing import Dict  # noqa

    duration_ms = 5_000.0
    rate_hz = 12.0
    noise_std = 0.3
    n_neurons = 3

    # Generate ground truth
    spike_trains = generate_ground_truth(
        n_neurons=n_neurons,
        rate_hz=rate_hz,
        duration_ms=duration_ms,
        seed=42,
    )

    # Build simple templates (biphasic AP shapes)
    dt_ms = 0.1
    t = np.linspace(-1.5, 1.5, 30)

    tpl_0 = np.array(
        [
            -8.0 * np.exp(-((tt - 0.1) ** 2) / 0.3)
            + 2.0 * np.exp(-((tt + 0.3) ** 2) / 0.5)
            for tt in t
        ]
    )
    tpl_1 = np.array(
        [
            -5.0 * np.exp(-((tt - 0.0) ** 2) / 0.5)
            + 1.5 * np.exp(-((tt + 0.5) ** 2) / 0.4)
            for tt in t
        ]
    )
    tpl_2 = np.array(
        [
            -3.0 * np.exp(-((tt - 0.2) ** 2) / 1.0)
            + 1.0 * np.exp(-((tt + 0.1) ** 2) / 0.8)
            for tt in t
        ]
    )

    templates = [tpl_0, tpl_1, tpl_2]

    # Create contaminated recording
    recording, true_assignments = contaminate_recording(
        spike_times_list=spike_trains,
        templates=templates,
        noise_std=noise_std,
        sampling_hz=10_000.0,
        duration_ms=duration_ms,
        seed=42,
    )

    # Sort
    detected_spikes, per_neuron_count = template_sort(
        recording=recording,
        templates=templates,
        dt_ms=dt_ms,
        threshold=0.3,
        refractory_ms=2.0,
        time_window_ms=1.5,
        max_iter=1000,
    )

    print(f"Detected {len(detected_spikes)} spikes across {n_neurons} neurons")
    for nid, count in per_neuron_count.items():  # type: ignore[arg-type]
        print(f"  Neuron {nid}: {count} spikes")

    # Evaluate
    report = evaluate_sorting(
        true_spikes=true_assignments,
        detected_spikes=detected_spikes,
        merge_window_ms=0.5,
    )

    print_sorting_report(report)
