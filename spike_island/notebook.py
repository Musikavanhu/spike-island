"""Spike Island — Interactive Notebook Generator.

Programmatic generation of a complete Jupyter notebook walkthrough
covering every module in the Spike Island toolkit.

Each cell is hand-crafted: markdown for explanations and theory,
code cells for live demonstrations with visualizations.

Usage
-----
    >>> from spike_island.notebook import generate_notebook, save_notebook
    >>> nb = generate_notebook()
    >>> save_notebook(nb, "spike_island_walkthrough.ipynb")

The notebook covers:
    - Introduction & installation
    - Simulators (Poisson, Refractory, Bursty, Rhythmic)
    - Analysis metrics (ISI, CV, autocorrelation, raster)
    - Waveform generation & visualization
    - Spike sorting via template matching
    - Wilson-Cowan neural oscillator
    - STDP synaptic learning
    - Full pipeline orchestration
    - Performance benchmarks
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import nbformat

# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------


def _md(source: str) -> dict[str, Any]:
    """Create a markdown cell."""
    cell = nbformat.v4.new_markdown_cell(textwrap.dedent(source))
    return {
        "cell_type": "markdown",
        "source": cell.source,
        "metadata": {},
        "attachments": {},
    }


def _code(source: str) -> dict[str, Any]:
    """Create a code cell."""
    cell = nbformat.v4.new_code_cell(textwrap.dedent(source))
    return {
        "cell_type": "code",
        "source": cell.source,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "execution_count": None,
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _title_section() -> list[dict[str, Any]]:
    """Title, intro, and quick-start cells."""
    return [
        _md(
            """
            # 🧠 Spike Island: Neural Spike Train Toolkit

            A complete walkthrough of the **Spike Island** neurotech simulation library.

            Each section below is executable — run the cells to explore
            neural spiking, signal analysis, spike sorting, and synaptic plasticity.

            ---

            ## 📦 Installation

            ```bash
            pip install -e .
            python -m spike_island           # full pipeline demo
            python -m spike_island --quick    # quick validation
            python -m spike_island --bench    # benchmark suite
            ```
            """,
        ),
        _md(
            """
            ## 🏗️ Library Structure

            | Module | Purpose |
            |---|---|
            | `simulators` | Generate spike trains (Poisson, Refractory, Bursty, Rhythmic) |
            | `analysis` | Compute ISI histograms, CV, autocorrelograms, raster plots |
            | `waveforms` | Build continuous-time extracellular waveforms with noise |
            | `sorting` | Template-matching spike sorter for multi-unit data |
            | `oscillator` | Wilson-Cowan E-I neural mass model (fixed points, oscillations, bifurcations) |
            | `stdp` | Spike-timing dependent plasticity learning engine |
            | `pipeline` | End-to-end orchestration of all modules |
            | `benchmarks` | Performance profiling suite |
            | `notebook` | This interactive walkthrough |
            """,
        ),
        _code(
            """
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.facecolor': '#1a1a2e',
                    'axes.facecolor': '#16213e',
                    'axes.edgecolor': '#0f3460',
                    'text.color': '#e2e8f0',
                    'figure.dpi': 120})

print("✅ Environment ready for Spike Island walkthrough")
            """,
        ),
    ]


def _simulators_section() -> list[dict[str, Any]]:
    """Simulators section — 4 generator types."""
    return [
        _md(
            """
            ## 🎯 1. Spike Train Generators

            Neural spiking is fundamentally a stochastic process. Different neurons
            exhibit different temporal patterns — from memoryless Poisson firing to
            rhythmic bursting.

            ### Four Firing Patterns

            | Pattern | Mechanism | CV Range | Biological analog |
            |---|---|---|---|
            | **Poisson** | Memoryless point process | ~1.0 | Background cortical firing |
            | **Refractory Poisson** | Absolute refractory period enforced | 0.6–0.9 | Regular cortical neurons |
            | **Bursty Poisson** | Poisson trigger → burst of spikes | 1.5–2.5 | Thalamic burst firing |
            | **Rhythmic** | Fixed period + Gaussian jitter | 0.0–0.5 | Pacer/oscillatory neurons |

            The **coefficient of variation (CV)** of inter-spike intervals (ISI)
            is the standard metric: CV = σ_ISI / μ_ISI.
            """,
        ),
        _code(
            """
from spike_island.simulators import (
    poisson_spikes,
    refractory_poisson,
    bursty_poisson,
    rhythmic_spikes,
)

# Generate 5-second spike trains for each pattern
duration_ms = 5_000.0
seed = 42

poisson = poisson_spikes(rate_hz=20.0, duration_ms=duration_ms, seed=seed)
refractory = refractory_poisson(rate_hz=20.0, refractory_ms=2.0,
                                  duration_ms=duration_ms, seed=seed)
bursty = bursty_poisson(trigger_rate_hz=5.0, burst_size_mean=4.0,
                          burst_cv=0.3, duration_ms=duration_ms, seed=seed)
rhythmic = rhythmic_spikes(period_ms=50.0, jitter_sigma_ms=2.0,
                             duration_ms=duration_ms, seed=seed)

print(f"Poisson:      {len(poisson):4d} spikes")
print(f"Refractory:   {len(refractory):4d} spikes")
print(f"Bursty:       {len(bursty):4d} spikes")
print(f"Rhythmic:     {len(rhythmic):4d} spikes")
            """,
        ),
        _code(
            """
fig, axes = plt.subplots(4, 1, figsize=(12, 8),
                          gridspec_kw={'height_ratios': [1, 1, 1, 1]})

labels = ['Poisson', 'Refractory Poisson', 'Bursty Poisson', 'Rhythmic']
trains = [poisson, refractory, bursty, rhythmic]

for ax, label, train in zip(axes, labels, trains):
    ax.eventplot(train, lineoffsets=np.arange(len(train)) % 20 + 1,
                  color='#e94560', linewidth=1.0)
    ax.set_ylim(0, 22)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_title(f'{label}  (n={len(train):,} spikes)', fontsize=11, color='#e2e8f0')
    ax.set_yticks([])
    ax.grid(True, alpha=0.1)

plt.tight_layout()
plt.savefig('plots/notebook_raster.png', bbox_inches='tight')
plt.show()
print("\\n📊 Raster plot saved → plots/notebook_raster.png")
            """,
        ),
    ]


def _analysis_section() -> list[dict[str, Any]]:
    """Analysis section — ISI, CV, autocorrelation."""
    return [
        _md(
            """
            ## 📊 2. Analysis Metrics

            Once we have spike trains, we quantify them:

            - **ISI histogram**: distribution of inter-spike intervals
            - **CV (Coefficient of Variation)**: σ_ISI / μ_ISI — irregularity index
            - **Autocorrelogram**: spike train auto-correlation revealing periodicity
            - **Firing rate**: spikes / second over the recording window

            Low CV → regular firing (e.g. rhythm generators).
            High CV → irregular or bursty firing.
            """,
        ),
        _code(
            """
from spike_island.analysis import (
    analyze,
    isi_histogram,
    coefficient_of_variation,
    autocorrelogram,
    mean_firing_rate,
)

# Analyze all four spike trains
results = {}
for name, train in [('Poisson', poisson),
                     ('Refractory', refractory),
                     ('Bursty', bursty),
                     ('Rhythmic', rhythmic)]:
    results[name] = analyze(train, duration_ms=duration_ms)
    print(f"{name:12s} | Rate: {results[name]['firing_rate_hz']:6.1f} Hz "
          f"| CV: {results[name]['cv']:6.3f} "
          f"| Median ISI: {results[name]['median_isi_ms']:6.1f} ms")
            """,
        ),
        _code(
            """
# ISI histograms for all four patterns
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axs = axes.ravel()

for idx, (name, train) in enumerate([
    ('Poisson', poisson),
    ('Refractory', refractory),
    ('Bursty', bursty),
    ('Rhythmic', rhythmic)
]):
    counts, edges = isi_histogram(train, bin_width_ms=2.0)
    ax = axs[idx]
    ax.bar(edges[:-1], counts, width=2.0, color='#e94560', alpha=0.8)
    cv = coefficient_of_variation(train)
    ax.set_title(f'{name}  (CV = {cv:.3f})', fontsize=11)
    ax.set_xlabel('ISI (ms)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_facecolor('#16213e')
    ax.xaxis.label.set_color('#e2e8f0')
    ax.yaxis.label.set_color('#e2e8f0')

plt.tight_layout()
plt.savefig('plots/notebook_isi.png', bbox_inches='tight')
plt.show()
            """,
        ),
        _code(
            """
# Autocorrelograms — reveals periodicity and refractory structure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axs = axes.ravel()

for idx, (name, train) in enumerate([
    ('Poisson', poisson),
    ('Refractory', refractory),
    ('Bursty', bursty),
    ('Rhythmic', rhythmic)
]):
    ac = autocorrelogram(train, bin_width_ms=1.0, max_lag_ms=200.0)
    ax = axs[idx]
    ax.plot(ac[0], ac[1], color='#e94560', linewidth=1.2)
    ax.set_title(f'{name} Autocorrelogram', fontsize=11)
    ax.set_xlabel('Lag (ms)', fontsize=9)
    ax.set_ylabel('Coincidence count', fontsize=9)
    ax.set_facecolor('#16213e')

plt.tight_layout()
plt.savefig('plots/notebook_autocorr.png', bbox_inches='tight')
plt.show()
            """,
        ),
    ]


def _waveforms_section() -> list[dict[str, Any]]:
    """Waveform generation and visualization."""
    return [
        _md(
            """
            ## 🌊 3. Waveform Generation

            Real extracellular electrodes record continuous voltage where each action
            potential appears as a brief (~1–3 ms) biphasic waveform:
            rapid negative downstroke followed by a slower positive undershoot.

            This module:
            - Generates realistic AP templates from parametric Gaussian components
            - Superimposes templates at spike times to build continuous traces
            - Adds configurable noise (Gaussian white or 1/f pink noise)
            - Renders multi-channel waveform dashboards
            """,
        ),
        _code(
            """
from spike_island.waveforms import (
    generate_ap_template,
    spikes_to_waveform,
    waveform_statistics,
    add_noise,
    render_waveform_dashboard,
)

# Generate a single AP template
t, template = generate_ap_template(
    duration_ms=2.0,
    peak_amplitude_mv=-3.0,
    undershoot_amplitude_mv=1.0,
)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t, template, color='#e94560', linewidth=2)
ax.axhline(0, color='#535353', linewidth=0.5)
ax.set_xlabel('Time (ms)', fontsize=11)
ax.set_ylabel('Voltage (mV)', fontsize=11)
ax.set_title('Single Action Potential Template', fontsize=13)
ax.set_facecolor('#16213e')
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig('plots/notebook_template.png', bbox_inches='tight')
plt.show()
            """,
        ),
        _code(
            """
# Build a continuous waveform from the Poisson spike train
dt_ms = 0.1  # 10 kHz sampling
sampling_hz = 10_000.0

t_continuous, waveform = spikes_to_waveform(
    spike_times=poisson,
    template=template,
    dt_ms=dt_ms,
    duration_ms=500.0,  # first 500 ms only for display
    noise_type='gaussian',
    noise_rms_mv=0.15,
    seed=42,
)

# Compute waveform statistics
stats = waveform_statistics(waveform)
print(f"Waveform stats: min={stats['min_amplitude_mv']:.2f} mV, "
      f"max={stats['max_amplitude_mv']:.2f} mV, "
      f"rms={stats['rms_amplitude_mv']:.2f} mV")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_continuous, waveform, color='#e94560', linewidth=0.7)
ax.set_xlabel('Time (ms)', fontsize=11)
ax.set_ylabel('Voltage (mV)', fontsize=11)
ax.set_title('Extracellular Waveform (Poisson, 500 ms)', fontsize=13)
ax.set_facecolor('#16213e')
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig('plots/notebook_waveform.png', bbox_inches='tight')
plt.show()
            """,
        ),
    ]


def _sorting_section() -> list[dict[str, Any]]:
    """Spike sorting via template matching."""
    return [
        _md(
            """
            ## 🔍 4. Spike Sorting

            **Problem**: A single electrode records superposed waveforms from multiple
            neurons. We must detect spike times and assign each to its source neuron.

            **Approach**: Template matching (matching pursuit):
            1. Learn templates from known or clustered waveforms
            2. Slide templates across the recording
            3. At each position, find the best-matching template
            4. Subtract the matched template and iterate (greedy pursuit)

            **Metrics**: Precision, recall, F1-score per neuron class.
            """,
        ),
        _code(
            """
from spike_island.sorting import (
    generate_ground_truth,
    contaminate_recording,
    template_sort,
    evaluate_sorting,
    render_sorting_dashboard,
)

# Generate 3 neurons with different firing rates
ground_truth = generate_ground_truth(
    n_neurons=3,
    rate_hz=15.0,
    duration_ms=5_000.0,
    seed=42,
)

print(f"Ground truth: {sum(len(t) for t in ground_truth):,} spikes across 3 neurons")
for i, spikes in enumerate(ground_truth):
    print(f"  Neuron {i}: {len(spikes)} spikes")

# Build contaminated recording (superposed + noise)
recording = contaminate_recording(
    spike_trains=ground_truth,
    dt_ms=0.1,
    duration_ms=5_000.0,
    noise_rms_mv=0.2,
    seed=42,
)
print(f"\\nRecording length: {len(recording[0]):,} samples ({len(recording[0]) * 0.1:.0f} ms)")
            """,
        ),
        _code(
            """
# Extract templates from ground truth (using first 20 spikes per neuron)
templates = []
for spikes in ground_truth:
    _, t = generate_ap_template(duration_ms=2.0, dt_ms=0.1)
    # Vary template slightly per neuron
    templates.append(t.copy())

# Run matching pursuit sorter
events = template_sort(
    recording=recording,
    templates=templates,
    dt_ms=0.1,
    threshold=0.5,
    min_gap_ms=1.0,
)
print(f"Detected {len(events)} spike events")

# Evaluate against ground truth
report = evaluate_sorting(events, ground_truth, tolerance_ms=0.3)
print(f"Overall: Precision={report.precision:.3f}, "
      f"Recall={report.recall:.3f}, F1={report.f1_score:.3f}")
            """,
        ),
    ]


def _oscillator_section() -> list[dict[str, Any]]:
    """Wilson-Cowan neural oscillator."""
    return [
        _md(
            """
            ## 🔄 5. Wilson-Cowan Neural Oscillator

            A **mean-field model** of two interacting neural populations:
            excitatory (E) and inhibitory (I).

            ### Core Equations

            ```
            dE/dt = (-E + S(w_EE·E - w_EI·I + I_E)) / τ_E
            dI/dt = (-I + S(w_IE·E - w_II·I + I_I)) / τ_I
            ```

            Where `S(x)` is a sigmoidal activation function.

            ### Key Phenomena

            - **Oscillations** (gamma rhythms): Strong E↔I coupling
            - **Bistability** (working memory): Strong recurrent excitation
            - **Fixed points**: Weak coupling → single stable equilibrium
            """,
        ),
        _code(
            """
from spike_island.oscillator import (
    simulate_wilson_cowan,
    find_fixed_points,
    render_wilson_cowan,
    bifurcation_scan,
    OscillatorParams,
)

# Simulate oscillatory regime
params = OscillatorParams(
    w_ee=3.0, w_ei=2.0,
    w_ie=2.0, w_ii=1.0,
    tau_e=80.0, tau_i=20.0,
    i_e=0.0, i_i=0.0,
)

result = simulate_wilson_cowan(
    params=params,
    t_max=500.0,
    dt=0.5,
    seed=42,
)

print(f"Simulated {len(result.time_ms)} time steps ({result.time_ms[-1]:.0f} ms)")
print(f"Final state: E={result.e[-1]:.3f}, I={result.i[-1]:.3f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(result.time_ms, result.e, color='#e94560', linewidth=1.0)
axes[0].set_ylabel('E(t)', fontsize=11)
axes[0].set_title('Wilson-Cowan Oscillator (500 ms)', fontsize=13)
axes[1].plot(result.time_ms, result.i, color='#0f3460', linewidth=1.0)
axes[1].set_ylabel('I(t)', fontsize=11)
axes[1].set_xlabel('Time (ms)', fontsize=11)
for ax in axes:
    ax.set_facecolor('#16213e')
    ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig('plots/notebook_wc.png', bbox_inches='tight')
plt.show()
            """,
        ),
        _code(
            """
# Bifurcation scan: vary w_EE and observe behavior
bc_result = bifurcation_scan(
    w_ee_range=np.linspace(1.0, 5.0, 80),
    w_ei=2.0, w_ie=2.0, w_ii=1.0,
    tau_e=80.0, tau_i=20.0,
    t_max=2000.0, dt=1.0,
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(bc_result.w_ee, bc_result.max_e, color='#e94560', linewidth=2)
ax.plot(bc_result.w_ee, bc_result.min_e, color='#535353', linewidth=1, linestyle='--')
ax.set_xlabel('w_EE (recurrent excitation)', fontsize=11)
ax.set_ylabel('E(t) range', fontsize=11)
ax.set_title('Bifurcation: E(t) vs w_EE', fontsize=13)
ax.set_facecolor('#16213e')
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig('plots/notebook_bifurcation.png', bbox_inches='tight')
plt.show()
            """,
        ),
    ]


def _stdp_section() -> list[dict[str, Any]]:
    """STDP synaptic plasticity."""
    return [
        _md(
            """
            ## 🧬 6. STDP (Spike-Timing Dependent Plasticity)

            Synaptic strength changes based on the **relative timing** of pre-
            and postsynaptic spikes:

            - **Causal** (pre → post): **LTP** (long-term potentiation) → strengthen
            - **Anti-causal** (post → pre): **LTD** (long-term depression) → weaken

            ### Exponential Learning Window

            ```
            Δw =  A⁺ · exp(-Δt/τ⁺)   if Δt > 0  (pre before post)
            Δw = -A⁻ · exp(Δt/τ⁻)    if Δt < 0  (post before pre)
            ```

            Where Δt = t_post - t_pre.
            """,
        ),
        _code(
            """
from spike_island.stdp import (
    STDPParams,
    stdp_learning_window,
    run_stdp,
    render_stdp,
)

# Plot the learning window
dts = np.linspace(-50, 50, 200)
dw = stdp_learning_window(
    dts,
    a_plus=0.01,
    a_minus=0.012,
    tau_plus=20.0,
    tau_minus=20.0,
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(dts, 0, dw, where=(dw >= 0),
                color='#e94560', alpha=0.7, label='LTP (strengthen)')
ax.fill_between(dts, dw, 0, where=(dw < 0),
                color='#0f3460', alpha=0.7, label='LTD (weaken)')
ax.axhline(0, color='#535353', linewidth=0.5)
ax.set_xlabel('Δt = t_post - t_pre (ms)', fontsize=11)
ax.set_ylabel('Δw (weight change)', fontsize=11)
ax.set_title('STDP Learning Window', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.set_facecolor('#16213e')
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig('plots/notebook_stdp_window.png', bbox_inches='tight')
plt.show()
            """,
        ),
        _code(
            """
# Run a full STDP simulation with paired pre/post spike trains
params = STDPParams(
    a_plus=0.01,
    a_minus=0.012,
    tau_plus=20.0,
    tau_minus=20.0,
    w_min=0.0,
    w_max=1.0,
    history_length=100.0,
)

pre_spikes = np.arange(0, 5000, 100)    # regular 100 Hz
post_spikes = np.arange(5, 5000, 100)    # 5 ms delay → causal → LTP

result = run_stdp(
    pre_spikes=pre_spikes,
    post_spikes=post_spikes,
    params=params,
    initial_weight=0.5,
)

print(f"Initial weight: {result.weights[0]:.4f}")
print(f"Final weight:   {result.weights[-1]:.4f}")
print(f"Net change:     {result.weights[-1] - result.weights[0]:+.4f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(result.step_indices, result.weights,
        color='#e94560', linewidth=1.5)
ax.set_xlabel('Learning step', fontsize=11)
ax.set_ylabel('Synaptic weight', fontsize=11)
ax.set_title('STDP Weight Evolution (pre→post, +5 ms)', fontsize=13)
ax.set_facecolor('#16213e')
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig('plots/notebook_stdp_evolution.png', bbox_inches='tight')
plt.show()
            """,
        ),
    ]


def _pipeline_section() -> list[dict[str, Any]]:
    """Full pipeline orchestration."""
    return [
        _md(
            """
            ## 🚀 7. Full Pipeline

            The `pipeline` module orchestrates all 6 core modules into a
            single reproducible workflow:

            ```
            Simulators → Analysis → Waveforms → Sorting → Wilson-Cowan → STDP
            ```

            It produces:
            - A **dashboard** (3×2 figure with all results)
            - A **text report** with all metrics
            - **Timing** for each stage
            """,
        ),
        _code(
            """
from spike_island.pipeline import PipelineConfig, run_pipeline, run_and_report

config = PipelineConfig(
    duration_ms=3_000.0,
    sampling_hz=10_000.0,
    seed=42,
    output_dir='plots',
    wc_t_max=500.0,
    stdp_steps=30,
)

result = run_pipeline(config)
print(f"Pipeline completed in {result.total_elapsed_ms:.1f} ms")
print(f"  Simulators:     {result.simulator_elapsed_ms:.1f} ms")
print(f"  Analysis:       {result.analysis_elapsed_ms:.1f} ms")
print(f"  Waveforms:      {result.waveform_elapsed_ms:.1f} ms")
print(f"  Sorting:        {result.sorting_elapsed_ms:.1f} ms")
print(f"  Wilson-Cowan:   {result.oscillator_elapsed_ms:.1f} ms")
print(f"  STDP:           {result.stdp_elapsed_ms:.1f} ms")
            """,
        ),
    ]


def _benchmarks_section() -> list[dict[str, Any]]:
    """Performance benchmarks."""
    return [
        _md(
            """
            ## ⏱️ 8. Performance Benchmarks

            The `benchmarks` module profiles every component:

            - Wall-clock timing (via `time.perf_counter`)
            - Peak memory (via `tracemalloc`)
            - Throughput metrics (spikes/s, Hz/s, etc.)

            Use these to detect regressions or compare hardware.
            """,
        ),
        _code(
            """
from spike_island.benchmarks import (
    run_benchmarks,
    print_benchmark_report,
    save_benchmark_report,
)

suite = run_benchmarks()
print_benchmark_report(suite)

save_benchmark_report(suite, 'plots/notebook_benchmarks.txt')
print("\\n✅ Benchmark report saved → plots/notebook_benchmarks.txt")
            """,
        ),
    ]


def _conclusion_section() -> list[dict[str, Any]]:
    """Conclusion and next steps."""
    return [
        _md(
            """
            ## 🎓 Summary

            You've explored the full Spike Island toolkit:

            | # | Module | What you learned |
            |---|---|---|
            | 1 | **Simulators** | 4 neural firing patterns and their ISI statistics |
            | 2 | **Analysis** | CV, ISI histograms, autocorrelograms |
            | 3 | **Waveforms** | From spike times → continuous extracellular voltage |
            | 4 | **Sorting** | Template-matching spike classification with precision/recall |
            | 5 | **Wilson-Cowan** | E-I dynamics, bifurcations, oscillations |
            | 6 | **STDP** | Synaptic weight evolution from spike timing |
            | 7 | **Pipeline** | End-to-end orchestration |
            | 8 | **Benchmarks** | Performance profiling |

            ### Next Steps

            - 🔬 Add **GLIF** (Generalized Integrate-and-Fire) neuron models
            - 🧩 Build **spatially embedded** networks with distance-dependent weights
            - 📊 Add **real data** imports (Neurodata Without Borders, MEA recordings)
            - 🤖 Implement **deep learning** spike sorters (SpyKING CIRCUS, Kilosort)
            - 🌐 Deploy as **interactive web app** (Streamlit/Gradio)

            ---

            _Built as a portfolio project to sharpen neurotech and BioMEMS skills._
            """
            """
            _Repository: [github.com/Musikavanhu/spike-island](https://github.com/Musikavanhu/spike-island)_
            """,
        ),
    ]


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def generate_notebook() -> nbformat.notebooknode.NotebookNode:
    """Generate the complete Spike Island walkthrough notebook.

    Returns
    -------
    nbformat.notebooknode.NotebookNode
        A valid Jupyter notebook (.ipynb v4) with all walkthrough sections.
    """
    nb = nbformat.v4.new_notebook()

    # Set metadata
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    }

    # Assemble all sections
    all_cells: list[dict[str, Any]] = []
    all_cells.extend(_title_section())
    all_cells.extend(_simulators_section())
    all_cells.extend(_analysis_section())
    all_cells.extend(_waveforms_section())
    all_cells.extend(_sorting_section())
    all_cells.extend(_oscillator_section())
    all_cells.extend(_stdp_section())
    all_cells.extend(_pipeline_section())
    all_cells.extend(_benchmarks_section())
    all_cells.extend(_conclusion_section())

    # Convert dicts to nbformat cells
    for cell_dict in all_cells:
        if cell_dict["cell_type"] == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(cell_dict["source"]))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(cell_dict["source"]))

    return nb


def save_notebook(
    nb: nbformat.notebooknode.NotebookNode,
    path: str | Path = "spike_island_walkthrough.ipynb",
) -> Path:
    """Save a notebook to disk.

    Parameters
    ----------
    nb : NotebookNode
        The notebook object (e.g., from ``generate_notebook()``).
    path : str or Path
        Output file path (default: ``spike_island_walkthrough.ipynb``).

    Returns
    -------
    Path
        Absolute path to the saved notebook.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out))
    return out


def generate_and_save(
    path: str | Path = "spike_island_walkthrough.ipynb",
) -> Path:
    """Generate and save the notebook in one call.

    Convenience wrapper around ``generate_notebook()`` + ``save_notebook()``.

    Parameters
    ----------
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        Absolute path to the saved notebook.
    """
    nb = generate_notebook()
    return save_notebook(nb, path)


if __name__ == "__main__":
    saved = generate_and_save()
    print(f"Notebook saved → {saved}")
    nb = nbformat.read(str(saved), as_version=4)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    md_cells = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(f"  Code cells: {code_cells}")
    print(f"  Markdown cells: {md_cells}")
    print(f"  Total cells: {len(nb.cells)}")
