"""Full Demo Pipeline — End-to-End Neurotech Workflow.

Orchestrates all Spike Island modules into a single reproducible pipeline:

    Simulators → Analysis → Waveforms → Sorting → Wilson-Cowan → STDP

Each stage produces typed results that feed into the next, with a unified
report and dashboard at the end.  Designed as both a demo entry-point and
a reference integration pattern for downstream projects.

Usage
-----
>>> from spike_island.pipeline import PipelineConfig, run_pipeline
>>> config = PipelineConfig(duration_ms=5_000)
>>> result = run_pipeline(config)
>>> print(result.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Internal imports — all existing modules
# ---------------------------------------------------------------------------
from spike_island.simulators import (
    poisson_spikes,
    refractory_poisson,
    bursty_poisson,
    rhythmic_spikes,
)
from spike_island.analysis import analyze
from spike_island.waveforms import (
    generate_ap_template,
    spikes_to_waveform,
    waveform_statistics,
)
from spike_island.sorting import (
    SortingReport,
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
from spike_island.stdp import STDPParams, Synapse

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full demo pipeline.

    Parameters
    ----------
    duration_ms : float
        Duration of spike train simulations in milliseconds.
    sampling_hz : float
        Sampling rate for waveform generation (Hz).
    seed : int
        Random seed for reproducibility across all stages.
    output_dir : Path | str
        Directory to write plots and reports.  Created if missing.
    noise_std_mv : float
        Gaussian noise standard deviation in millivolts for waveforms.
    sort_threshold : float
        Correlation threshold for template matching spike sorter.
    wc_t_max : float
        Total Wilson-Cowan simulation time (ms).
    stdp_steps : int
        Number of STDP learning steps per synapse pair.
    """

    duration_ms: float = 5_000.0
    sampling_hz: float = 10_000.0
    seed: int = 42
    output_dir: Path | str = "plots"
    noise_std_mv: float = 0.05
    sort_threshold: float = 0.3
    wc_t_max: float = 3_000.0
    stdp_steps: int = 100

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


# ===================================================================
# Stage results
# ===================================================================

@dataclass
class StageResult:
    """Named result from a single pipeline stage."""
    name: str
    data: dict[str, Any]
    elapsed_ms: float = 0.0


@dataclass
class PipelineResult:
    """Aggregated results from all pipeline stages.

    Parameters
    ----------
    config : PipelineConfig
        The configuration used for this run.
    stages : list[StageResult]
        Ordered list of stage outputs.
    total_elapsed_ms : float
        Wall-clock time for the full pipeline in milliseconds.
    """

    config: PipelineConfig
    stages: list[StageResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0

    def add_stage(self, stage: StageResult) -> None:
        """Append a completed stage result."""
        self.stages.append(stage)

    def get_stage(self, name: str) -> Optional[StageResult]:
        """Retrieve a stage by name."""
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def summary(self) -> str:
        """Return a human-readable pipeline summary."""
        lines = [
            "=" * 60,
            "SPIKE ISLAND — Pipeline Summary",
            "=" * 60,
        ]
        for stage in self.stages:
            elapsed_str = f"{stage.elapsed_ms:.1f} ms"
            key_metrics = _extract_key_metrics(stage)
            lines.append(f"[{elapsed_str:>8s}] {stage.name}: {key_metrics}")
        lines.append("-" * 60)
        lines.append(f"Total pipeline time: {self.total_elapsed_ms:.1f} ms")
        lines.append("=" * 60)
        return "\n".join(lines)


# ===================================================================
# Helper utilities
# ===================================================================

def _extract_key_metrics(stage: StageResult) -> str:
    """Pull the most important metric(s) from a stage result."""
    d = stage.data
    if stage.name == "simulators":
        counts = {n: len(t) for n, t in d["spike_trains"].items()}
        return ", ".join(f"{k}={v}" for k, v in counts.items())
    if stage.name == "analysis":
        metrics_strs = []
        for r in d["analyses"]:
            metrics_strs.append(
                f"{r['name']}: CV={r['cv']:.3f}, rate={r['firing_rate_hz']:.1f}Hz"
            )
        return "; ".join(metrics_strs)
    if stage.name == "waveforms":
        stats = d.get("statistics", [])
        parts = [f"{s['name']}: RMS={s['rms_mv']:.3f}mV, SNR={s['snr_db']:.1f}dB" for s in stats]
        return "; ".join(parts) if parts else "no statistics"
    if stage.name == "sorting":
        r = d["report"]
        return f"precision={r.precision:.2f}, recall={r.recall:.2f}, f1={r.f1_score:.2f}"
    if stage.name == "wilson_cowan":
        m = d.get("metrics", {})
        return (
            f"oscillating={m.get('oscillating', False)}, "
            f"freq={m.get('frequency_hz', 0):.1f}Hz, "
            f"amp={m.get('amplitude', 0):.3f}"
        )
    if stage.name == "stdp":
        w = d["final_weight"]
        return f"w_initial={d['initial_weight']:.4f}, w_final={w:.4f}"
    return str(d)


def _time_block(func, *args, **kwargs) -> tuple[Any, float]:
    """Run *func* and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1_000.0
    return result, elapsed_ms


# ===================================================================
# Pipeline stages
# ===================================================================

def stage_simulators(config: PipelineConfig) -> StageResult:
    """Generate spike trains for all firing patterns."""
    logger.info("Stage: simulators — generating spike trains")

    def run():
        return {
            "Poisson": poisson_spikes(
                rate_hz=20.0, duration_ms=config.duration_ms, seed=config.seed
            ),
            "Refractory": refractory_poisson(
                rate_hz=20.0, duration_ms=config.duration_ms,
                refractory_ms=2.0, seed=config.seed,
            ),
            "Bursty": bursty_poisson(
                background_rate_hz=10.0,
                burst_rate=5.0,
                burst_size_mean=4,
                burst_size_std=1.5,
                intra_burst_isi_ms=5.0,
                duration_ms=config.duration_ms,
                seed=config.seed,
            ),
            "Rhythmic": rhythmic_spikes(
                rate_hz=10.0, duration_ms=config.duration_ms,
                jitter_sd_ms=1.0, seed=config.seed,
            ),
        }

    spike_trains, elapsed = _time_block(run)
    return StageResult(name="simulators", data={"spike_trains": spike_trains}, elapsed_ms=elapsed)


def stage_analysis(
    config: PipelineConfig, spike_trains: dict[str, np.ndarray]
) -> StageResult:
    """Compute analysis metrics for each spike train."""
    logger.info("Stage: analysis — computing ISI, CV, autocorrelation")

    def run():
        return [
            analyze(spikes, config.duration_ms, name=name)
            for name, spikes in spike_trains.items()
        ]

    analyses, elapsed = _time_block(run)
    return StageResult(name="analysis", data={"analyses": analyses}, elapsed_ms=elapsed)


def stage_waveforms(
    config: PipelineConfig,
    spike_trains: dict[str, np.ndarray],
    analyses: list[dict],
) -> StageResult:
    """Generate continuous-time waveforms and compute statistics."""
    logger.info("Stage: waveforms — generating AP templates & traces")

    template_t, template = generate_ap_template()

    def run():
        stats_list = []
        for analysis in analyses:
            times, voltage = spikes_to_waveform(
                analysis["spike_times"],
                template,
                template_dt=0.1,
                duration_ms=config.duration_ms,
                sampling_hz=config.sampling_hz,
                noise_std_mv=config.noise_std_mv,
                noise_type="gaussian",
                seed=config.seed,
            )
            stats = waveform_statistics(voltage, times)
            stats["name"] = analysis["name"]
            stats_list.append(stats)
        return {
            "template_t": template_t,
            "template": template,
            "statistics": stats_list,
        }

    wf_data, elapsed = _time_block(run)
    return StageResult(name="waveforms", data=wf_data, elapsed_ms=elapsed)


def stage_sorting(
    config: PipelineConfig, spike_trains: dict[str, np.ndarray]
) -> StageResult:
    """Run template-matching spike sorting on a synthetic multi-unit recording."""
    logger.info("Stage: sorting — template matching pursuit")

    spikes_0 = spike_trains["Poisson"]
    spikes_1 = poisson_spikes(
        rate_hz=15.0, duration_ms=config.duration_ms, seed=config.seed + 99
    )

    tpl_0 = np.array([-5.0, 0.0, 2.5, -0.3])
    tpl_1 = np.array([-3.0, 0.5, 0.0])

    def run():
        recording, true_assignments = contaminate_recording(
            spike_times_list=[spikes_0, spikes_1],
            templates=[tpl_0, tpl_1],
            noise_std=0.3,
            sampling_hz=config.sampling_hz,
            duration_ms=config.duration_ms,
            seed=config.seed,
        )

        detected, counts = template_sort(
            recording=recording,
            templates=[tpl_0, tpl_1],
            dt_ms=0.1,
            threshold=config.sort_threshold,
            refractory_ms=2.0,
            time_window_ms=1.0,
            max_iter=500,
        )

        report = evaluate_sorting(
            true_spikes=true_assignments,
            detected_spikes=detected,
            merge_window_ms=0.5,
        )
        return {
            "report": report,
            "recording_length": len(recording),
            "total_detected": sum(counts.values()),
        }

    sort_data, elapsed = _time_block(run)
    return StageResult(name="sorting", data=sort_data, elapsed_ms=elapsed)


def stage_wilson_cowan(config: PipelineConfig) -> StageResult:
    """Simulate Wilson-Cowan neural mass model and analyze dynamics."""
    logger.info("Stage: wilson_cowan — E-I oscillator simulation")

    wc_params = WCParams(
        tau_E=80.0,
        tau_I=20.0,
        w_EE=1.0,
        w_EI=1.0,
        w_IE=0.8,
        w_II=1.0,
        I_E=0.4,
        I_I=0.1,
    )

    def run():
        # Fixed point analysis
        fps = wc_fixed_points(wc_params)
        fp_info: list[dict] = []
        for fp in fps:
            stab = wc_stability(fp, wc_params)
            fp_info.append({
                "e_star": float(fp[0]),
                "i_star": float(fp[1]),
                **stab,
            })

        # Time simulation — returns (t, trajectory) where trajectory is (N, 2)
        t, traj = simulate_wc(
            wc_params, t_max=config.wc_t_max, dt=0.5, seed=config.seed
        )

        # Oscillation metrics — use transient proportional to sim duration
        transient_ms = min(config.wc_t_max * 0.4, 1000.0)
        metrics = wc_oscillation_metrics(t, traj, transient_ms=transient_ms)

        # Bifurcation scan — returns (N_vals, 4) array: [param_val, E_min, E_max, E_mean]
        param_values = np.linspace(0.4, 2.0, 30)
        bif_array = wc_bifurcation_scan(
            wc_params, vary_param="w_EE", param_values=param_values,
            t_max=config.wc_t_max, transient_ms=transient_ms, dt=0.5, seed=config.seed,
        )

        return {
            "fixed_points": fp_info,
            "times_ms": t,
            "trajectory": traj,
            "metrics": metrics,
            "bifurcation_array": bif_array,
            "params": wc_params,
        }

    wc_data, elapsed = _time_block(run)
    return StageResult(name="wilson_cowan", data=wc_data, elapsed_ms=elapsed)


def stage_stdp(config: PipelineConfig) -> StageResult:
    """Run STDP learning on a single synapse with paired spike trains."""
    logger.info("Stage: stdp — spike-timing dependent plasticity")

    params = STDPParams(
        a_plus=0.01,
        a_minus=0.012,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0,
        history_length=50.0,
    )

    rng = np.random.default_rng(config.seed)

    def run():
        synapse = Synapse(weight=0.5, params=params)

        # Causal pairing: pre fires before post → LTP expected
        n_steps = config.stdp_steps
        for _ in range(n_steps):
            pre_spike = rng.uniform(0.0, 10.0)
            post_spike = pre_spike + rng.uniform(1.0, 5.0)  # causal offset
            synapse.record_pre_spike(pre_spike)
            synapse.record_post_spike(post_spike)

        return {
            "initial_weight": 0.5,
            "final_weight": synapse.weight,
            "n_steps": n_steps,
            "weight_history": list(synapse.weight_history),
        }

    stdp_data, elapsed = _time_block(run)
    return StageResult(name="stdp", data=stdp_data, elapsed_ms=elapsed)


# ===================================================================
# Visualization
# ===================================================================

def plot_pipeline_dashboard(
    result: PipelineResult, output_path: Path | str | None = None
) -> Path:
    """Generate a unified 3×2 dashboard combining all pipeline stages.

    Parameters
    ----------
    result : PipelineResult
        Completed pipeline results.
    output_path : Path | str | None
        File path for the saved figure.  Defaults to
        ``output_dir/pipeline_dashboard.png``.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    if output_path is None:
        output_path = result.config.output_dir / "pipeline_dashboard.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Spike Island — Full Pipeline Dashboard", fontsize=16, fontweight="bold")

    # --- Panel (0,0): Spike raster from simulators ---
    ax_raster = axes[0, 0]
    _plot_raster(ax_raster, result)

    # --- Panel (0,1): ISI histograms from analysis ---
    ax_isi = axes[0, 1]
    _plot_isi_comparison(ax_isi, result)

    # --- Panel (1,0): Waveform traces ---
    ax_wf = axes[1, 0]
    _plot_waveform_sample(ax_wf, result)

    # --- Panel (1,1): Sorting confusion matrix ---
    ax_sort = axes[1, 1]
    _plot_sorting_report(ax_sort, result)

    # --- Panel (2,0): Wilson-Cowan timeseries ---
    ax_wc = axes[2, 0]
    _plot_wc_timeseries(ax_wc, result)

    # --- Panel (2,1): STDP weight evolution ---
    ax_stdp = axes[2, 1]
    _plot_stdp_result(ax_stdp, result)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Dashboard saved to %s", output_path)
    return output_path


def _plot_raster(ax: matplotlib.axes.Axes, result: PipelineResult) -> None:
    """Plot spike raster for all generator types."""
    sim_stage = result.get_stage("simulators")
    if not sim_stage:
        ax.text(0.5, 0.5, "No simulator data", ha="center", va="center",
                transform=ax.transAxes)
        return

    trains = sim_stage.data["spike_trains"]
    colors = {"Poisson": "#e74c3c", "Refractory": "#3498db",
              "Bursty": "#2ecc71", "Rhythmic": "#f39c12"}

    for idx, (name, spikes) in enumerate(trains.items()):
        y = np.full_like(spikes, idx, dtype=float)
        ax.scatter(
            spikes, y + 0.5, s=3, color=colors.get(name, "#888"),
            alpha=0.7, label=name,
        )

    ax.set_yticks(list(range(len(trains))))
    ax.set_yticklabels(trains.keys())
    ax.set_xlabel("Time (ms)")
    ax.set_title("Spike Raster — All Generators")


def _plot_isi_comparison(ax: matplotlib.axes.Axes, result: PipelineResult) -> None:
    """Overlay ISI histograms for all spike trains."""
    analysis_stage = result.get_stage("analysis")
    if not analysis_stage:
        ax.text(0.5, 0.5, "No analysis data", ha="center", va="center",
                transform=ax.transAxes)
        return

    analyses = analysis_stage.data["analyses"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i, a in enumerate(analyses):
        isis = np.diff(a["spike_times"])
        if len(isis) == 0:
            continue
        ax.hist(
            isis[isis < 500], bins=40, alpha=0.6,
            label=f"{a['name']} (CV={a['cv']:.2f})",
            color=colors[i % len(colors)], density=True,
        )

    ax.set_xlabel("ISI (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Inter-Spike Interval Distributions")
    ax.legend(fontsize=8)


def _plot_waveform_sample(ax: matplotlib.axes.Axes, result: PipelineResult) -> None:
    """Plot a sample waveform trace."""
    wf_stage = result.get_stage("waveforms")
    if not wf_stage:
        ax.text(0.5, 0.5, "No waveform data", ha="center", va="center",
                transform=ax.transAxes)
        return

    template = wf_stage.data["template"]
    t = np.arange(len(template)) * 0.1  # ms
    ax.plot(t, template, color="#9b59b6", linewidth=1.5, label="AP Template")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Action Potential Template")
    ax.legend()


def _plot_sorting_report(ax: matplotlib.axes.Axes, result: PipelineResult) -> None:
    """Plot sorting metrics as a bar chart."""
    sort_stage = result.get_stage("sorting")
    if not sort_stage:
        ax.text(0.5, 0.5, "No sorting data", ha="center", va="center",
                transform=ax.transAxes)
        return

    report = sort_stage.data["report"]
    metrics = ["Precision", "Recall", "F1 Score"]
    values = [report.precision, report.recall, report.f1_score]
    colors_bar = ["#3498db", "#e74c3c", "#2ecc71"]

    bars = ax.bar(metrics, values, color=colors_bar, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.1)
    ax.set_title("Spike Sorting Performance")


def _plot_wc_timeseries(ax: matplotlib.axes.Axes, result: PipelineResult) -> None:
    """Plot Wilson-Cowan E and I traces."""
    wc_stage = result.get_stage("wilson_cowan")
    if not wc_stage:
        ax.text(0.5, 0.5, "No WC data", ha="center", va="center",
                transform=ax.transAxes)
        return

    t_ms = wc_stage.data["times_ms"] / 1_000.0  # convert to seconds
    traj = wc_stage.data["trajectory"]
    ax.plot(t_ms, traj[:, 0], color="#e74c3c", label="Excitatory (E)", linewidth=0.8)
    ax.plot(t_ms, traj[:, 1], color="#3498db", label="Inhibitory (I)", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")
    ax.set_title("Wilson-Cowan Dynamics")
    ax.legend(fontsize=8)


def _plot_stdp_result(ax: matplotlib.axes.Axes, result: PipelineResult) -> None:
    """Plot STDP weight change summary."""
    stdp_stage = result.get_stage("stdp")
    if not stdp_stage:
        ax.text(0.5, 0.5, "No STDP data", ha="center", va="center",
                transform=ax.transAxes)
        return

    d = stdp_stage.data
    w_init = d["initial_weight"]
    w_final = d["final_weight"]
    delta = w_final - w_init

    ax.bar(["Initial", "Final"], [w_init, w_final],
           color=["#95a5a6", "#2ecc71"], width=0.4)
    ax.set_ylim(0, 1.1)
    direction = "LTP ↑" if delta > 0 else "LTD ↓"
    ax.set_title(f"STDP Weight Change ({direction}, Δ={delta:+.4f})")


# ===================================================================
# Report generation
# ===================================================================

def generate_text_report(result: PipelineResult) -> str:
    """Generate a comprehensive text report from pipeline results.

    Parameters
    ----------
    result : PipelineResult
        Completed pipeline output.

    Returns
    -------
    str
        Multi-line text report suitable for logging or saving to file.
    """
    import spike_island

    lines = [
        "=" * 70,
        "SPIKE ISLAND — Full Pipeline Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Version: spike-island v{spike_island.__version__}",
        "=" * 70,
        "",
    ]

    # Configuration summary
    cfg = result.config
    lines.append("## Configuration")
    lines.append(f"  Duration:       {cfg.duration_ms:.0f} ms")
    lines.append(f"  Sampling rate:  {cfg.sampling_hz:.0f} Hz")
    lines.append(f"  Seed:           {cfg.seed}")
    lines.append(f"  Noise σ (mV):   {cfg.noise_std_mv}")
    lines.append(f"  Sort threshold: {cfg.sort_threshold}")
    lines.append("")

    # Stage-by-stage details
    for stage in result.stages:
        lines.append(f"## [{stage.elapsed_ms:.1f} ms] {stage.name.upper()}")
        d = stage.data

        if stage.name == "simulators":
            for name, spikes in d["spike_trains"].items():
                lines.append(f"  {name:>12s}: {len(spikes):>5d} spikes")

        elif stage.name == "analysis":
            for a in d["analyses"]:
                lines.append(
                    f"  {a['name']:>12s}: rate={a['firing_rate_hz']:.1f} Hz, "
                    f"CV={a['cv']:.3f}, n_ISI={len(a.get('isi', []))}"
                )

        elif stage.name == "waveforms":
            for s in d["statistics"]:
                lines.append(
                    f"  {s['name']:>12s}: RMS={s['rms_mv']:.3f} mV, "
                    f"SNR={s['snr_db']:.1f} dB"
                )

        elif stage.name == "sorting":
            r = d["report"]
            lines.append(f"  Detected:     {r.total_detected}")
            lines.append(f"  Precision:    {r.precision:.4f}")
            lines.append(f"  Recall:       {r.recall:.4f}")
            lines.append(f"  F1 Score:     {r.f1_score:.4f}")

        elif stage.name == "wilson_cowan":
            m = d.get("metrics", {})
            lines.append(f"  Oscillating:  {m.get('oscillating', 'N/A')}")
            lines.append(f"  Frequency:    {m.get('frequency_hz', 'N/A')} Hz")
            lines.append(f"  Amplitude:    {m.get('amplitude', 'N/A')}")
            lines.append(f"  E mean:       {m.get('e_mean', 'N/A')}")
            bif = d["bifurcation_array"]
            osc_count = int(np.sum(bif[:, 2] - bif[:, 1] > 0.05))
            lines.append(
                f"  Bifurcation:   {osc_count}/{len(bif)} oscillatory points"
            )

        elif stage.name == "stdp":
            delta = d["final_weight"] - d["initial_weight"]
            direction = "LTP (strengthening)" if delta > 0 else "LTD (weakening)"
            lines.append(f"  Initial w:    {d['initial_weight']:.4f}")
            lines.append(f"  Final w:      {d['final_weight']:.4f}")
            lines.append(f"  Δw:           {delta:+.4f} ({direction})")

        lines.append("")

    # Footer
    lines.append("-" * 70)
    lines.append(f"Total pipeline time: {result.total_elapsed_ms:.1f} ms")
    lines.append("=" * 70)

    return "\n".join(lines)


# ===================================================================
# Main orchestrator
# ===================================================================

def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Execute the full Spike Island pipeline end-to-end.

    Parameters
    ----------
    config : PipelineConfig | None
        Pipeline configuration.  Uses defaults if ``None``.

    Returns
    -------
    PipelineResult
        Aggregated results from all stages, including timing and metrics.
    """
    if config is None:
        config = PipelineConfig()

    logger.info("Starting Spike Island pipeline (seed=%d)", config.seed)
    t_start = time.perf_counter()

    result = PipelineResult(config=config)

    # Stage 1: Simulators
    sim_stage = stage_simulators(config)
    result.add_stage(sim_stage)

    # Stage 2: Analysis
    analysis_stage = stage_analysis(config, sim_stage.data["spike_trains"])
    result.add_stage(analysis_stage)

    # Stage 3: Waveforms
    wf_stage = stage_waveforms(
        config, sim_stage.data["spike_trains"], analysis_stage.data["analyses"]
    )
    result.add_stage(wf_stage)

    # Stage 4: Sorting
    sort_stage = stage_sorting(config, sim_stage.data["spike_trains"])
    result.add_stage(sort_stage)

    # Stage 5: Wilson-Cowan
    wc_stage = stage_wilson_cowan(config)
    result.add_stage(wc_stage)

    # Stage 6: STDP
    stdp_stage = stage_stdp(config)
    result.add_stage(stdp_stage)

    result.total_elapsed_ms = (time.perf_counter() - t_start) * 1_000.0

    logger.info("Pipeline complete in %.1f ms", result.total_elapsed_ms)
    return result


def run_and_report(
    config: PipelineConfig | None = None,
    save_dashboard: bool = True,
    print_summary: bool = True,
) -> PipelineResult:
    """Run the pipeline and produce all outputs (dashboard + text report).

    Convenience wrapper that calls ``run_pipeline``, generates the dashboard
    plot, prints a summary, and saves a text report.

    Parameters
    ----------
    config : PipelineConfig | None
        Pipeline configuration.
    save_dashboard : bool
        Whether to generate and save the visualization dashboard.
    print_summary : bool
        Whether to print the pipeline summary to stdout.

    Returns
    -------
    PipelineResult
        The completed pipeline result object.
    """
    if config is None:
        config = PipelineConfig()

    result = run_pipeline(config)

    # Save dashboard plot
    if save_dashboard:
        dashboard_path = plot_pipeline_dashboard(result)
        logger.info("Dashboard saved to %s", dashboard_path)

    # Generate and optionally print text report
    report_text = generate_text_report(result)
    if print_summary:
        print(report_text)

    # Save text report to file
    report_path = config.output_dir / "pipeline_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    logger.info("Text report saved to %s", report_path)

    return result


# ===================================================================
# Demo entry point
# ===================================================================

def demo() -> None:
    """Run the full pipeline with default settings and print results."""
    logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")
    config = PipelineConfig(
        duration_ms=5_000.0,
        sampling_hz=10_000.0,
        seed=42,
        output_dir="plots",
    )
    run_and_report(config)
