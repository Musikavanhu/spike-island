"""Spike Island — Demo Entry Point.

Run ``python -m spike_island`` to execute the full pipeline demo:

    1. Generate spike trains (Poisson, Refractory, Bursty, Rhythmic)
    2. Compute analysis metrics (ISI, CV, autocorrelation)
    3. Build continuous-time waveforms with noise injection
    4. Run template-matching spike sorting on synthetic multi-unit data
    5. Simulate Wilson-Cowan neural mass dynamics
    6. Learn synaptic weights via STDP

Outputs:
- ``plots/pipeline_dashboard.png`` — 3×2 visualization dashboard
- ``plots/pipeline_report.txt``   — Text report with all metrics

Usage
-----
>>> python -m spike_island              # full pipeline demo (default)
>>> python -m spike_island --quick      # shorter run for quick validation
"""

from __future__ import annotations

import argparse
import logging
import sys

from spike_island.pipeline import PipelineConfig, run_and_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Spike Island — Full neurotech pipeline demo",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a shorter simulation for quick validation (~200 ms)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots",
        help="Directory to save plots and reports (default: plots/)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full pipeline demo."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s [%(levelname)s] %(message)s",
    )

    if args.quick:
        config = PipelineConfig(
            duration_ms=200.0,
            sampling_hz=1_000.0,
            seed=args.seed,
            output_dir=args.output_dir,
            wc_t_max=300.0,
            stdp_steps=20,
        )
    else:
        config = PipelineConfig(
            duration_ms=5_000.0,
            sampling_hz=10_000.0,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    result = run_and_report(config)
    print(f"\n✅ Pipeline complete in {result.total_elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
