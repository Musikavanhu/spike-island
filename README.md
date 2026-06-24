# spike-island
A minimal spike train simulator and analyzer for learning neurophysiology fundamentals.
Simulates neural spiking activity with different firing patterns, computes standard neurophysiology metrics, and visualizes results.
## 🧠 What this is
**Day-by-day neurotech portfolio project** — each commit adds a meaningful module. Built to sharpen skills for BioMEMS and neural signal processing roles.
## ⚡ Quick start
```bash
pip install -e .
python -m spike_island
```
This generates spike trains, computes analysis metrics, and saves a dashboard to `plots/analysis_dashboard.png`.
## 📐 Generators
| Generator | Firing pattern | Typical CV |
|---|---|---|
| `Poisson` | Memoryless, regular | ~1.0 |
| `Refractory Poisson` | Regular with absolute refractory period | 0.6–0.9 |
| `Bursty Poisson` | Irregular bursts on Poisson background | 1.5–2.5 |
| `Rhythmic` | Regular with Gaussian jitter | 0.0–0.5 |
## 📦 Modules
- `simulators` — Generate spike trains (Day 1)
- `analysis` — ISI histograms, firing rate, CV, autocorrelation (Day 2)
- `waveforms` — AP templates, continuous voltage traces, waveform dashboard (Day 3)
## 🚧 Daily milestones
| Day | Module | Status |
|---|---|---|
| 1 | Core simulators | ✅ |
| 2 | Analysis metrics (ISI, CV, autocorrelation) | ✅ |
| 3 | Waveform visualization dashboard | ✅ |
| 4 | Spike sorting (template matching) | ⏳ |
| 5 | Wilson-Cowan neural oscillator | ⏳ |
| 6 | Full demo pipeline | ⏳ |
---
Built by [Tino](https://github.com/Musikavanhu) — daily commits to build a neurotech portfolio.
