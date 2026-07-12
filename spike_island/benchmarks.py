"""Spike Island — Performance Profiling & Benchmarking Suite.

Provides systematic benchmarks for every module in the Spike Island toolkit:

    - Simulators: wall-clock time vs. duration and rate
    - Analysis: wall-clock time vs. spike count
    - Waveforms: wall-clock time vs. sampling rate and spike count
    - Sorting: wall-clock time vs. recording length
    - Wilson-Cowan: wall-clock time vs. simulation length
    - STDP: wall-clock time vs. step count
    - Full pipeline: end-to-end wall-clock time

Each benchmark returns a ``BenchmarkResult`` with timing, memory, and
throughput metrics.  The suite can be run standalone or integrated into
CI/CD pipelines for regression detection.

Usage
-----
>>> from spike_island.benchmarks import run_benchmarks, print_benchmark_report
>>> results = run_benchmarks()
>>> print_benchmark_report(results)

References
----------
- ``time.perf_counter`` for high-resolution wall-clock timing
- ``tracemalloc`` for peak memory allocation tracking
- NumPy vectorized operations vs. Python loops (refractory Poisson)
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from spike_island.simulators import (
    poisson_spikes,
    refractory_poisson,
    bursty_poisson,
    rhythmic_spikes,
)
from spike_island.analysis import (
    analyze,
    coefficient_of_variation,
    autocorrelogram,
    isi_histogram,
    mean_firing_rate,
)
from spike_island.waveforms import (
    generate_ap_template,
    spikes_to_waveform,
    waveform_statistics,
)
from spike_island.sorting import (
    contaminate_recording,
    template_sort,
    evaluate_sorting,
)
from spike_island.oscillator import (
    WCParams,
    simulate_wc,
    wc_fixed_points,
    wc_stability,
    wc_bifurcation_scan,
    wc_oscillation_metrics,
)
from spike_island.stdp import (
    STDPParams,
    Synapse,
    STDPNetwork,
)
from spike_island.pipeline import (
    PipelineConfig,
    run_pipeline,
)


# ===================================================================
# Data types
# ===================================================================


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes
    ----------
    name : str
        Human-readable benchmark name (e.g. "simulators.poisson_10s").
    category : str
        Module category (e.g. "simulators", "analysis").
    elapsed_ms : float
        Wall-clock time in milliseconds.
    peak_memory_mb : float
        Peak memory allocation during the benchmark (MB), via ``tracemalloc``.
    throughput : float
        Throughput metric (e.g. spikes/ms, Hz/ms).  Interpretation depends
        on the benchmark; stored as a generic float.
    metadata : dict[str, Any]
        Additional context (input sizes, parameters, etc.).
    """

    name: str
    category: str
    elapsed_ms: float
    peak_memory_mb: float
    throughput: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Aggregated results from a full benchmark suite run.

    Attributes
    ----------
    results : list[BenchmarkResult]
        All individual benchmark results, in execution order.
    total_elapsed_ms : float
        Wall-clock time for the entire suite.
    """

    results: list[BenchmarkResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0

    def add(self, result: BenchmarkResult) -> None:
        """Append a benchmark result."""
        self.results.append(result)

    def by_category(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by category."""
        groups: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            groups.setdefault(r.category, []).append(r)
        return groups


# ===================================================================
# Benchmark runner utility
# ===================================================================


def _run_benchmark(
    name: str,
    category: str,
    func: Callable[[], Any],
    throughput_key: str = "items",
    throughput_value: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Execute *func* with timing and memory profiling.

    Parameters
    ----------
    name : str
        Benchmark name.
    category : str
        Module category.
    func : Callable
        Zero-argument function to benchmark.
    throughput_key : str
        Label for the throughput metric (e.g. "spikes/ms").
    throughput_value : float
        Numerator for throughput (e.g. number of spikes generated).
    metadata : dict or None
        Extra context to store in the result.

    Returns
    -------
    BenchmarkResult
        Timed and profiled result.
    """
    # Start memory tracking
    tracemalloc.start()

    t0 = time.perf_counter()
    output = func()
    elapsed_ms = (time.perf_counter() - t0) * 1_000.0

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_bytes / (1024 * 1024)

    # Compute throughput: items per ms
    throughput = throughput_value / elapsed_ms if elapsed_ms > 0 else 0.0

    return BenchmarkResult(
        name=name,
        category=category,
        elapsed_ms=elapsed_ms,
        peak_memory_mb=peak_memory_mb,
        throughput=throughput,
        metadata=metadata or {},
    )


# ===================================================================
# Simulator benchmarks
# ===================================================================


def benchmark_simulators() -> list[BenchmarkResult]:
    """Benchmark all spike train generators across multiple scales.

    Tests each generator at short (1 s), medium (10 s), and long (60 s)
    durations to capture scaling behavior.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each generator at each scale.
    """
    results: list[BenchmarkResult] = []
    durations = [1_000.0, 10_000.0, 60_000.0]
    seed = 42

    for dur in durations:
        dur_label = f"{dur / 1000:.0f}s"

        # Poisson
        def _poisson():
            return poisson_spikes(rate_hz=50.0, duration_ms=dur, seed=seed)

        spikes = _poisson()
        results.append(
            _run_benchmark(
                name=f"simulators.poisson_{dur_label}",
                category="simulators",
                func=_poisson,
                throughput_key="spikes/ms",
                throughput_value=len(spikes),
                metadata={"duration_ms": dur, "rate_hz": 50.0, "spikes": len(spikes)},
            )
        )

        # Refractory Poisson
        def _refractory():
            return refractory_poisson(
                rate_hz=50.0, duration_ms=dur, refractory_ms=2.0, seed=seed
            )

        spikes = _refractory()
        results.append(
            _run_benchmark(
                name=f"simulators.refractory_{dur_label}",
                category="simulators",
                func=_refractory,
                throughput_key="spikes/ms",
                throughput_value=len(spikes),
                metadata={"duration_ms": dur, "rate_hz": 50.0, "spikes": len(spikes)},
            )
        )

        # Bursty Poisson
        def _bursty():
            return bursty_poisson(
                background_rate_hz=30.0,
                burst_rate=10.0,
                burst_size_mean=5,
                burst_size_std=2.0,
                intra_burst_isi_ms=3.0,
                duration_ms=dur,
                seed=seed,
            )

        spikes = _bursty()
        results.append(
            _run_benchmark(
                name=f"simulators.bursty_{dur_label}",
                category="simulators",
                func=_bursty,
                throughput_key="spikes/ms",
                throughput_value=len(spikes),
                metadata={"duration_ms": dur, "spikes": len(spikes)},
            )
        )

        # Rhythmic
        def _rhythmic():
            return rhythmic_spikes(
                rate_hz=30.0, duration_ms=dur, jitter_sd_ms=1.0, seed=seed
            )

        spikes = _rhythmic()
        results.append(
            _run_benchmark(
                name=f"simulators.rhythmic_{dur_label}",
                category="simulators",
                func=_rhythmic,
                throughput_key="spikes/ms",
                throughput_value=len(spikes),
                metadata={"duration_ms": dur, "rate_hz": 30.0, "spikes": len(spikes)},
            )
        )

    return results


# ===================================================================
# Analysis benchmarks
# ===================================================================


def benchmark_analysis() -> list[BenchmarkResult]:
    """Benchmark analysis functions across multiple spike counts.

    Generates spike trains of varying sizes and benchmarks:
    - ``analyze()`` (full analysis pipeline)
    - ``coefficient_of_variation()``
    - ``autocorrelogram()``
    - ``isi_histogram()``

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each analysis function at each scale.
    """
    results: list[BenchmarkResult] = []
    seed = 42

    # Generate spike trains of different sizes
    sizes = [
        (100, 1_000.0),
        (1_000, 10_000.0),
        (10_000, 100_000.0),
    ]

    for count, dur in sizes:
        size_label = f"{count:,}"
        # Generate a spike train with approximately `count` spikes
        rate = count / (dur / 1000.0)
        spikes = poisson_spikes(rate_hz=rate, duration_ms=dur, seed=seed)

        # Full analyze
        def _analyze():
            return analyze(spikes, dur, name="bench")

        _analyze()
        results.append(
            _run_benchmark(
                name=f"analysis.analyze_{size_label}",
                category="analysis",
                func=_analyze,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={"spike_count": len(spikes), "duration_ms": dur},
            )
        )

        # CV
        def _cv():
            return coefficient_of_variation(spikes)

        _cv()
        results.append(
            _run_benchmark(
                name=f"analysis.cv_{size_label}",
                category="analysis",
                func=_cv,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={"spike_count": len(spikes)},
            )
        )

        # Autocorrelogram
        def _acg():
            return autocorrelogram(spikes, bin_width_ms=5.0)

        _acg()
        results.append(
            _run_benchmark(
                name=f"analysis.autocorrelogram_{size_label}",
                category="analysis",
                func=_acg,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={"spike_count": len(spikes)},
            )
        )

        # ISI histogram
        def _isi():
            return isi_histogram(spikes, bin_width_ms=5.0)

        _isi()
        results.append(
            _run_benchmark(
                name=f"analysis.isi_histogram_{size_label}",
                category="analysis",
                func=_isi,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={"spike_count": len(spikes)},
            )
        )

    return results


# ===================================================================
# Waveform benchmarks
# ===================================================================


def benchmark_waveforms() -> list[BenchmarkResult]:
    """Benchmark waveform generation at multiple scales.

    Tests ``spikes_to_waveform()`` and ``waveform_statistics()`` at
    varying sampling rates and spike counts.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for waveform operations.
    """
    results: list[BenchmarkResult] = []
    seed = 42
    template_t, template = generate_ap_template()

    configs = [
        {"spike_count": 50, "sampling_hz": 10_000.0, "dur_ms": 1_000.0},
        {"spike_count": 200, "sampling_hz": 30_000.0, "dur_ms": 5_000.0},
        {"spike_count": 1_000, "sampling_hz": 50_000.0, "dur_ms": 10_000.0},
    ]

    for cfg in configs:
        label = f"{cfg['spike_count']:,}spk_{cfg['sampling_hz'] / 1000:.0f}kHz"
        # Generate spike times
        rng = np.random.default_rng(seed)
        spike_times = np.sort(rng.uniform(0, cfg["dur_ms"], size=cfg["spike_count"]))

        def _waveform(st=spike_times, tpl=template, **kw):
            return spikes_to_waveform(
                st,
                tpl,
                template_dt=0.1,
                duration_ms=kw["dur_ms"],
                sampling_hz=kw["sampling_hz"],
                noise_std_mv=0.05,
                noise_type="gaussian",
                seed=seed,
            )

        times, voltage = _waveform(dur_ms=cfg["dur_ms"], sampling_hz=cfg["sampling_hz"])

        def _stats():
            return waveform_statistics(voltage, times)

        results.append(
            _run_benchmark(
                name=f"waveforms.to_waveform_{label}",
                category="waveforms",
                func=lambda: _waveform(dur_ms=cfg["dur_ms"], sampling_hz=cfg["sampling_hz"]),
                throughput_key="samples/ms",
                throughput_value=len(voltage),
                metadata=cfg,
            )
        )

        results.append(
            _run_benchmark(
                name=f"waveforms.statistics_{label}",
                category="waveforms",
                func=_stats,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata=cfg,
            )
        )

    return results


# ===================================================================
# Sorting benchmarks
# ===================================================================


def benchmark_sorting() -> list[BenchmarkResult]:
    """Benchmark spike sorting at multiple recording lengths.

    Generates synthetic multi-unit recordings and benchmarks
    ``template_sort()`` and ``evaluate_sorting()``.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for sorting operations.
    """
    results: list[BenchmarkResult] = []
    seed = 42

    configs = [
        {"dur_ms": 100.0, "rate_hz": 10.0},
        {"dur_ms": 200.0, "rate_hz": 15.0},
        {"dur_ms": 500.0, "rate_hz": 20.0},
    ]

    for cfg in configs:
        label = f"{cfg['dur_ms'] / 1000:.0f}s"
        # Generate two neuron spike trains
        spikes_0 = poisson_spikes(cfg["rate_hz"], cfg["dur_ms"], seed)
        spikes_1 = poisson_spikes(cfg["rate_hz"] * 0.8, cfg["dur_ms"], seed + 1)

        tpl_0 = np.array([-5.0, 0.0, 2.5, -0.3])
        tpl_1 = np.array([-3.0, 0.5, 0.0])

        recording, true_assignments = contaminate_recording(
            spike_times_list=[spikes_0, spikes_1],
            templates=[tpl_0, tpl_1],
            noise_std=0.3,
            sampling_hz=10_000.0,
            duration_ms=cfg["dur_ms"],
            seed=seed,
        )

        def _sort():
            return template_sort(
                recording=recording,
                templates=[tpl_0, tpl_1],
                dt_ms=0.1,
                threshold=0.3,
                refractory_ms=2.0,
                time_window_ms=1.0,
                max_iter=50,
            )

        detected, counts = _sort()
        total_detected = sum(counts.values())

        results.append(
            _run_benchmark(
                name=f"sorting.template_sort_{label}",
                category="sorting",
                func=_sort,
                throughput_key="samples/ms",
                throughput_value=len(recording),
                metadata={
                    "recording_length": len(recording),
                    "detected": total_detected,
                    "duration_ms": cfg["dur_ms"],
                },
            )
        )

        def _evaluate():
            return evaluate_sorting(
                true_spikes=true_assignments,
                detected_spikes=detected,
                merge_window_ms=0.5,
            )

        results.append(
            _run_benchmark(
                name=f"sorting.evaluate_{label}",
                category="sorting",
                func=_evaluate,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={"true_count": len(true_assignments), "detected": total_detected},
            )
        )

    return results


# ===================================================================
# Wilson-Cowan benchmarks
# ===================================================================


def benchmark_oscillator() -> list[BenchmarkResult]:
    """Benchmark Wilson-Cowan simulation and analysis at multiple scales.

    Tests ``simulate_wc()``, ``wc_fixed_points()``, ``wc_stability()``,
    ``wc_bifurcation_scan()``, and ``wc_oscillation_metrics()``.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for oscillator operations.
    """
    results: list[BenchmarkResult] = []
    seed = 42

    wc_params = WCParams(
        tau_E=80.0, tau_I=20.0,
        w_EE=1.0, w_EI=1.0,
        w_IE=0.8, w_II=1.0,
        I_E=0.4, I_I=0.1,
    )

    t_max_values = [1_000.0, 5_000.0, 20_000.0]

    for t_max in t_max_values:
        label = f"{t_max / 1000:.0f}s"

        def _simulate():
            return simulate_wc(wc_params, t_max=t_max, dt=0.5, seed=seed)

        t, traj = _simulate()

        results.append(
            _run_benchmark(
                name=f"oscillator.simulate_{label}",
                category="oscillator",
                func=_simulate,
                throughput_key="timesteps/ms",
                throughput_value=len(t),
                metadata={"t_max_ms": t_max, "timesteps": len(t)},
            )
        )

        # Fixed points
        def _fp():
            return wc_fixed_points(wc_params)

        fps = _fp()

        results.append(
            _run_benchmark(
                name=f"oscillator.fixed_points_{label}",
                category="oscillator",
                func=_fp,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={"fixed_points_found": len(fps)},
            )
        )

        # Stability analysis
        def _stability():
            return [wc_stability(fp, wc_params) for fp in fps]

        results.append(
            _run_benchmark(
                name=f"oscillator.stability_{label}",
                category="oscillator",
                func=_stability,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={"fixed_points": len(fps)},
            )
        )

        # Oscillation metrics
        transient_ms = min(t_max * 0.4, 1000.0)

        def _metrics():
            return wc_oscillation_metrics(t, traj, transient_ms=transient_ms)

        results.append(
            _run_benchmark(
                name=f"oscillator.metrics_{label}",
                category="oscillator",
                func=_metrics,
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={"t_max_ms": t_max, "transient_ms": transient_ms},
            )
        )

    # Bifurcation scan
    param_values = np.linspace(0.4, 2.0, 50)

    def _bifurcation():
        return wc_bifurcation_scan(
            wc_params, vary_param="w_EE", param_values=param_values,
            t_max=5_000.0, transient_ms=1000.0, dt=0.5, seed=seed,
        )

    results.append(
        _run_benchmark(
            name="oscillator.bifurcation_scan_50pts",
            category="oscillator",
            func=_bifurcation,
            throughput_key="param_values/ms",
            throughput_value=len(param_values),
            metadata={"param_count": len(param_values)},
        )
    )

    return results


# ===================================================================
# STDP benchmarks
# ===================================================================


def benchmark_stdp() -> list[BenchmarkResult]:
    """Benchmark STDP learning at multiple scales.

    Tests single-synapse and network-level STDP.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for STDP operations.
    """
    results: list[BenchmarkResult] = []
    seed = 42
    params = STDPParams(
        a_plus=0.01, a_minus=0.012,
        tau_plus=20.0, tau_minus=20.0,
        w_min=0.0, w_max=1.0, history_length=50.0,
    )
    rng = np.random.default_rng(seed)

    step_counts = [10, 100, 500]

    for n_steps in step_counts:
        label = f"{n_steps:,}steps"

        def _single_synapse():
            syn = Synapse(weight=0.5, params=params)
            for _ in range(n_steps):
                pre_t = rng.uniform(0.0, 10.0)
                post_t = pre_t + rng.uniform(1.0, 5.0)
                syn.record_pre_spike(pre_t)
                syn.record_post_spike(post_t)
            return syn.weight

        _single_synapse()

        results.append(
            _run_benchmark(
                name=f"stdp.single_synapse_{label}",
                category="stdp",
                func=_single_synapse,
                throughput_key="steps/ms",
                throughput_value=n_steps,
                metadata={"steps": n_steps},
            )
        )

    # Network benchmarks
    network_sizes = [(5, 5), (20, 20), (50, 50)]

    for n_pre, n_post in network_sizes:
        label = f"{n_pre}x{n_post}"

        def _network(np_=n_pre, np__=n_post):
            net = STDPNetwork(
                n_pre=np_, n_post=np__,
                params=params, w_init=0.5, history_length=50.0,
            )
            # Generate random spike events
            rng_net = np.random.default_rng(seed)
            for _ in range(200):
                pre_neuron = rng_net.integers(0, np_)
                pre_t = rng_net.uniform(0.0, 100.0)
                net.record_pre_spike(int(pre_neuron), pre_t)
                post_neuron = rng_net.integers(0, np__)
                post_t = rng_net.uniform(0.0, 100.0)
                net.record_post_spike(int(post_neuron), post_t)
            return net.get_stats()

        _network()

        results.append(
            _run_benchmark(
                name=f"stdp.network_{label}",
                category="stdp",
                func=_network,
                throughput_key="synapses/ms",
                throughput_value=n_pre * n_post,
                metadata={"n_pre": n_pre, "n_post": n_post, "synapses": n_pre * n_post},
            )
        )

    return results


# ===================================================================
# Full pipeline benchmark
# ===================================================================


def benchmark_pipeline() -> list[BenchmarkResult]:
    """Benchmark the full Spike Island pipeline at multiple scales.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for the full pipeline.
    """
    results: list[BenchmarkResult] = []
    seed = 42

    configs = [
        PipelineConfig(duration_ms=200.0, sampling_hz=5_000.0, seed=seed,
                       wc_t_max=200.0, stdp_steps=10),
        PipelineConfig(duration_ms=500.0, sampling_hz=5_000.0, seed=seed,
                       wc_t_max=500.0, stdp_steps=20),
    ]

    for cfg in configs:
        label = f"{cfg.duration_ms / 1000:.0f}s"

        def _run_pipeline():
            return run_pipeline(cfg)

        result = _run_pipeline()

        results.append(
            _run_benchmark(
                name=f"pipeline.full_{label}",
                category="pipeline",
                func=lambda: _run_pipeline(),
                throughput_key="runs/ms",
                throughput_value=1.0,
                metadata={
                    "duration_ms": cfg.duration_ms,
                    "stages": len(result.stages),
                    "total_elapsed_ms": result.total_elapsed_ms,
                },
            )
        )

    return results


# ===================================================================
# Suite orchestrator
# ===================================================================


def run_benchmarks() -> BenchmarkSuite:
    """Run the full benchmark suite across all modules.

    Executes benchmarks for simulators, analysis, waveforms, sorting,
    oscillator, STDP, and the full pipeline.  Returns aggregated results
    with timing and memory profiling.

    Returns
    -------
    BenchmarkSuite
        All benchmark results with total suite execution time.
    """
    t_start = time.perf_counter()
    suite = BenchmarkSuite()

    for bench_func in [
        benchmark_simulators,
        benchmark_analysis,
        benchmark_waveforms,
        benchmark_sorting,
        benchmark_oscillator,
        benchmark_stdp,
        benchmark_pipeline,
    ]:
        for result in bench_func():
            suite.add(result)

    suite.total_elapsed_ms = (time.perf_counter() - t_start) * 1_000.0
    return suite


# ===================================================================
# Report generation
# ===================================================================


def print_benchmark_report(suite: BenchmarkSuite) -> str:
    """Format and print a benchmark report.

    Parameters
    ----------
    suite : BenchmarkSuite
        Completed benchmark suite.

    Returns
    -------
    str
        The formatted report text.
    """
    lines = [
        "=" * 80,
        "SPIKE ISLAND — Benchmark Report",
        "=" * 80,
        f"Total suite time: {suite.total_elapsed_ms:.1f} ms",
        f"Benchmarks run:   {len(suite.results)}",
        "",
    ]

    by_cat = suite.by_category()
    for category in sorted(by_cat.keys()):
        results = by_cat[category]
        lines.append(f"{'─' * 80}")
        lines.append(f"  {category.upper()}")
        lines.append(f"{'─' * 80}")
        lines.append(
            f"  {'Name':<45s} {'Time (ms)':>10s} {'Memory (MB)':>12s} {'Throughput':>12s}"
        )
        lines.append(f"  {'─' * 45} {'─' * 10} {'─' * 12} {'─' * 12}")

        for r in results:
            lines.append(
                f"  {r.name:<45s} {r.elapsed_ms:>10.2f} {r.peak_memory_mb:>12.2f} "
                f"{r.throughput:>12.4f}"
            )

        lines.append("")

    lines.append("=" * 80)
    report = "\n".join(lines)
    print(report)
    return report


def save_benchmark_report(suite: BenchmarkSuite, path: str = "plots/benchmark_report.txt") -> None:
    """Save benchmark report to a text file.

    Parameters
    ----------
    suite : BenchmarkSuite
        Completed benchmark suite.
    path : str
        Output file path.
    """
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    report = print_benchmark_report(suite)
    # print_benchmark_report already prints; we re-generate for file
    lines = [
        "=" * 80,
        "SPIKE ISLAND — Benchmark Report",
        "=" * 80,
        f"Total suite time: {suite.total_elapsed_ms:.1f} ms",
        f"Benchmarks run:   {len(suite.results)}",
        "",
    ]

    by_cat = suite.by_category()
    for category in sorted(by_cat.keys()):
        results = by_cat[category]
        lines.append(f"{'─' * 80}")
        lines.append(f"  {category.upper()}")
        lines.append(f"{'─' * 80}")
        lines.append(
            f"  {'Name':<45s} {'Time (ms)':>10s} {'Memory (MB)':>12s} {'Throughput':>12s}"
        )
        lines.append(f"  {'─' * 45} {'─' * 10} {'─' * 12} {'─' * 12}")

        for r in results:
            lines.append(
                f"  {r.name:<45s} {r.elapsed_ms:>10.2f} {r.peak_memory_mb:>12.2f} "
                f"{r.throughput:>12.4f}"
            )
        lines.append("")

    lines.append("=" * 80)
    p.write_text("\n".join(lines))
