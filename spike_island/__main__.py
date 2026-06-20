"""Allow running with: python3 -m spike_island"""

from spike_island.simulators import (
    bursty_poisson,
    poisson_spikes,
    refractory_poisson,
    rhythmic_spikes,
)
from spike_island.analysis import analyze, plot_analysis_dashboard

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
