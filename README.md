# spike-island
A minimal spike train simulator and analyzer for learning neurophysiology fundamentals.
Simulates neural spiking activity with different firing patterns, computes standard neurophysiology metrics, visualizes results, and sorts spike events from multi-unit recordings.

## 🧠 What this is
**Day-by-day neurotech portfolio project** — each commit adds a meaningful module. Built to sharpen skills for BioMEMS and neural signal processing roles.

## ⚡ Quick start
```bash
pip install -e .
python -m spike_island
```
This generates a spike train visualization and saves it to `plots/spike_train_demo.png`.

## 📐 Generators
| Generator | Firing pattern | Typical CV |
|---|---|---|
| `Poisson` | Memoryless, regular | ~1.0 |
| `Refractory Poisson` | Regular with absolute refractory period | 0.6–0.9 |
| `Bursty Poisson` | Irregular bursts on Poisson background | 1.5–2.5 |
| `Rhythmic` | Regular with Gaussian jitter | 0.0–0.5 |

## 📦 Modules
- `simulators` — Generate spike trains
- `analysis` — ISI histograms, firing rate, CV, autocorrelation (Day 2)
- `waveforms` — Continuous-time AP templates & spike waveform pipeline (Day 3)
- `sorting` — Template matching spike sorter via matching pursuit (Day 4)
- `wilson_cowan` — Mean-field E-I oscillator: fixed points, stability, bifurcation scans, oscillation metrics (Day 5)
- `stdp` — Spike-timing dependent plasticity with exponential learning rules (Day 6)
- `pipeline` — Full demo pipeline orchestrating all modules into a unified workflow (Day 7)
- `benchmarks` — Performance profiling & benchmarking suite for all modules (Day 8)

## 🚧 Daily milestones
| Day | Module | Status |
|---|---|---|
| 1 | Core simulators | ✅ |
| 2 | Analysis metrics (ISI, CV, autocorrelation) | ✅ |
| 3 | Waveform visualization dashboard | ✅ |
| 4 | Spike sorting (template matching) | ✅ |
| 5 | Wilson-Cowan neural oscillator | ✅ |
| 6 | STDP learning engine | ✅ |
| 7 | Full demo pipeline | ✅ |
| 8 | Performance profiling & benchmarking suite | ✅ |

---
Built by [Tino](https://github.com/Musikavanhu) — daily commits to build a neurotech portfolio.
