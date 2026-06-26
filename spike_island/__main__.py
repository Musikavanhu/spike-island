"""Allow running with: python3 -m spike_island"""

from spike_island.simulators import (
    bursty_poisson,
    poisson_spikes,
    refractory_poisson,
    rhythmic_spikes,
)
from spike_island.analysis import analyze, plot_analysis_dashboard
from spike_island.waveforms import (
    generate_ap_template,
    plot_waveform_dashboard,
    spikes_to_waveform,
    waveform_statistics,
)
import numpy as np

# Generate spike trains
duration_ms = 5_000.0  # 5 seconds

trains = {
    "Poisson": poisson_spikes(rate_hz=20.0, duration_ms=duration_ms, seed=42),
    "Refractory": refractory_poisson(
        rate_hz=20.0, duration_ms=duration_ms, refractory_ms=2.0, seed=42
    ),
    "Bursty": bursty_poisson(
        background_rate_hz=10.0,
        burst_rate=5.0,
        burst_size_mean=4,
        burst_size_std=1.5,
        intra_burst_isi_ms=5.0,
        duration_ms=duration_ms,
        seed=42,
    ),
    "Rhythmic": rhythmic_spikes(rate_hz=10.0, duration_ms=duration_ms, jitter_sd_ms=1.0, seed=42),
}

# Analyze each
results = [analyze(spikes, duration_ms, name=name) for name, spikes in trains.items()]

# Print metrics
for res in results:
    print(f"  {res['name']:>10s}  spikes={res['spikes']:>5d}  "
          f"CV={res['cv']:.3f}  rate={res['firing_rate_hz']:.1f} Hz")

# Save analysis dashboard
plot_analysis_dashboard(results, duration_ms)

# --- Day 3: Waveform visualization ---
template_t, template = generate_ap_template()

# Generate and display waveform traces for each neuron
waveform_results = []
for res in results:
    times, voltage = spikes_to_waveform(
        res["spike_times"],
        template,
        template_dt=0.1,
        duration_ms=duration_ms,
        sampling_hz=10_000.0,
        noise_std_mv=0.05,
        noise_type="gaussian",
        seed=42,
    )
    stats = waveform_statistics(voltage, times)
    waveform_results.append({
        "name": res["name"],
        "spike_times": res["spike_times"],
        "rms_mv": stats["rms_mv"],
        "snr_db": stats["snr_db"],
    })
    print(f"  {res['name']:>10s}  RMS={stats['rms_mv']:.3f} mV  SNR={stats['snr_db']:.1f} dB")

# Save waveform dashboard
plot_waveform_dashboard(
    waveform_results,
    duration_ms=duration_ms,
    template=template,
    template_dt=0.1,
    sampling_hz=10_000.0,
    noise_std_mv=0.05,
    seed=42,
)

# --- Day 4: Spike sorting ---
from spike_island.sorting import (
    contaminate_recording,
    evaluate_sorting,
    generate_ground_truth,
    print_sorting_report,
    template_sort,
)

# Use the first neuron's spike times as a simplified sorting test
test_spikes = trains["Poisson"]
tpl_0 = np.array([-5.0, 0.0, 2.5, -0.3])
tpl_1 = np.array([-3.0, 0.5, 0.0])

# Generate a second neuron's spike train
test_spikes_1 = poisson_spikes(rate_hz=15.0, duration_ms=duration_ms, seed=99)

recording, true_assignments = contaminate_recording(
    spike_times_list=[test_spikes, test_spikes_1],
    templates=[tpl_0, tpl_1],
    noise_std=0.3,
    sampling_hz=10_000.0,
    duration_ms=duration_ms,
    seed=42,
)

detected, counts = template_sort(
    recording=recording,
    templates=[tpl_0, tpl_1],
    dt_ms=0.1,
    threshold=0.3,
    refractory_ms=2.0,
    time_window_ms=1.0,
    max_iter=500,
)

report = evaluate_sorting(
    true_spikes=true_assignments,
    detected_spikes=detected,
    merge_window_ms=0.5,
)

print(f"  {'':>10s}  {'Sorting:':>12s} detected={report.total_detected}  "
      f"precision={report.precision:.2f}  recall={report.recall:.2f}  f1={report.f1_score:.2f}")
