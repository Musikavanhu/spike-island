"""Tests for spike_island.pipeline — Full Demo Pipeline."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from spike_island.pipeline import (
    PipelineConfig,
    PipelineResult,
    StageResult,
    _extract_key_metrics,
    _time_block,
    generate_text_report,
    plot_pipeline_dashboard,
    run_and_report,
    run_pipeline,
    stage_analysis,
    stage_simulators,
    stage_sorting,
    stage_stdp,
    stage_waveforms,
    stage_wilson_cowan,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def config() -> PipelineConfig:
    """Short-duration config for fast tests."""
    return PipelineConfig(
        duration_ms=500.0,
        sampling_hz=1_000.0,
        seed=42,
        output_dir=tempfile.mkdtemp(),
        wc_t_max=500.0,
        stdp_steps=20,
    )


# ===================================================================
# Configuration tests
# ===================================================================

class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.duration_ms == 5_000.0
        assert cfg.sampling_hz == 10_000.0
        assert cfg.seed == 42
        assert isinstance(cfg.output_dir, Path)

    def test_custom_output_dir(self) -> None:
        cfg = PipelineConfig(output_dir="/tmp/test_plots")
        assert cfg.output_dir == Path("/tmp/test_plots")


# ===================================================================
# StageResult / PipelineResult tests
# ===================================================================

class TestStageResult:
    def test_creation(self) -> None:
        sr = StageResult(name="test", data={"key": "val"}, elapsed_ms=10.5)
        assert sr.name == "test"
        assert sr.data["key"] == "val"
        assert sr.elapsed_ms == 10.5


class TestPipelineResult:
    def test_add_and_get_stage(self, config) -> None:
        result = PipelineResult(config=config)
        stage = StageResult(name="sim", data={"x": 1}, elapsed_ms=5.0)
        result.add_stage(stage)
        assert result.get_stage("sim") is stage
        assert result.get_stage("nonexistent") is None

    def test_summary(self, config) -> None:
        result = PipelineResult(config=config)
        result.add_stage(StageResult(name="simulators", data={
            "spike_trains": {"Poisson": np.array([1.0, 2.0])}
        }, elapsed_ms=5.0))
        summary = result.summary()
        assert "SPIKE ISLAND" in summary
        assert "simulators" in summary


# ===================================================================
# Helper function tests
# ===================================================================

class TestExtractKeyMetrics:
    def test_simulators(self) -> None:
        sr = StageResult(name="simulators", data={
            "spike_trains": {"Poisson": np.array([1, 2, 3])}
        })
        assert "Poisson=3" in _extract_key_metrics(sr)

    def test_analysis(self) -> None:
        sr = StageResult(name="analysis", data={
            "analyses": [{"name": "A", "cv": 0.5, "firing_rate_hz": 10.0}]
        })
        assert "CV=0.500" in _extract_key_metrics(sr)

    def test_sorting(self) -> None:
        from spike_island.sorting import SortingReport
        report = SortingReport(
            total_detected=90, matched=80, false_positives=10,
            false_negatives=20, precision=0.89, recall=0.80, f1_score=0.84,
        )
        sr = StageResult(name="sorting", data={"report": report})
        metrics = _extract_key_metrics(sr)
        assert "precision" in metrics.lower()

    def test_unknown_stage(self) -> None:
        sr = StageResult(name="unknown", data={"x": 1})
        assert isinstance(_extract_key_metrics(sr), str)


class TestTimeBlock:
    def test_returns_result_and_time(self) -> None:
        result, elapsed = _time_block(lambda: 42)
        assert result == 42
        assert elapsed >= 0.0


# ===================================================================
# Stage-level tests
# ===================================================================

class TestStageSimulators:
    def test_returns_all_four_generators(self, config) -> None:
        stage = stage_simulators(config)
        trains = stage.data["spike_trains"]
        assert set(trains.keys()) == {"Poisson", "Refractory", "Bursty", "Rhythmic"}

    def test_spike_counts_reasonable(self, config) -> None:
        stage = stage_simulators(config)
        # 500 ms at ~20 Hz ≈ 10 spikes for Poisson
        assert len(stage.data["spike_trains"]["Poisson"]) > 0

    def test_spikes_sorted(self, config) -> None:
        stage = stage_simulators(config)
        for name, spikes in stage.data["spike_trains"].items():
            assert np.all(np.diff(spikes) >= 0), f"{name} not sorted"

    def test_timing_recorded(self, config) -> None:
        stage = stage_simulators(config)
        assert stage.elapsed_ms > 0.0


class TestStageAnalysis:
    def test_returns_analyses_for_each_train(self, config) -> None:
        trains = {
            "Poisson": np.array([10.0, 50.0, 90.0]),
            "Rhythmic": np.array([20.0, 60.0, 100.0]),
        }
        stage = stage_analysis(config, trains)
        assert len(stage.data["analyses"]) == 2

    def test_analysis_has_cv(self, config) -> None:
        trains = {"Test": np.array([10.0, 50.0, 90.0])}
        stage = stage_analysis(config, trains)
        assert "cv" in stage.data["analyses"][0]


class TestStageWaveforms:
    def test_returns_template_and_statistics(self, config) -> None:
        spike_times = np.array([10.0, 50.0, 90.0])
        trains = {"Test": spike_times}
        analyses = [{"name": "Test", "spike_times": spike_times}]

        stage = stage_waveforms(config, trains, analyses)
        assert "template" in stage.data
        assert len(stage.data["statistics"]) == 1
        assert "rms_mv" in stage.data["statistics"][0]


class TestStageSorting:
    def test_returns_report(self, config) -> None:
        trains = {
            "Poisson": np.array([10.0, 50.0, 90.0]),
        }
        stage = stage_sorting(config, trains)
        assert isinstance(stage.data["report"], type(stage.data["report"]))
        assert hasattr(stage.data["report"], "precision")


class TestStageWilsonCowan:
    def test_returns_trajectory(self, config) -> None:
        stage = stage_wilson_cowan(config)
        traj = stage.data["trajectory"]
        assert traj.ndim == 2
        assert traj.shape[1] == 2  # E and I columns

    def test_returns_metrics(self, config) -> None:
        stage = stage_wilson_cowan(config)
        m = stage.data["metrics"]
        assert "oscillating" in m
        assert "frequency_hz" in m


class TestStageSTDP:
    def test_weight_changes_with_causal_pairing(self, config) -> None:
        stage = stage_stdp(config)
        # Causal pairing should produce LTP (weight increase)
        delta = stage.data["final_weight"] - stage.data["initial_weight"]
        assert delta > 0.0, "Causal pairing should strengthen synapse"

    def test_weight_bounded(self, config) -> None:
        stage = stage_stdp(config)
        assert 0.0 <= stage.data["final_weight"] <= 1.0


# ===================================================================
# Full pipeline tests
# ===================================================================

class TestRunPipeline:
    def test_returns_all_six_stages(self, config) -> None:
        result = run_pipeline(config)
        assert len(result.stages) == 6
        names = [s.name for s in result.stages]
        expected = ["simulators", "analysis", "waveforms", "sorting",
                     "wilson_cowan", "stdp"]
        assert names == expected

    def test_total_elapsed_ms(self, config) -> None:
        result = run_pipeline(config)
        assert result.total_elapsed_ms > 0.0

    def test_default_config_works(self) -> None:
        # Uses default output_dir which may not exist — should still work
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PipelineConfig(output_dir=tmpdir, duration_ms=200.0, wc_t_max=200.0)
            result = run_pipeline(cfg)
            assert len(result.stages) == 6


class TestRunAndReport:
    def test_saves_dashboard(self, config) -> None:
        result = run_and_report(config, print_summary=False)
        dashboard_path = Path(config.output_dir) / "pipeline_dashboard.png"
        assert dashboard_path.exists()

    def test_saves_text_report(self, config) -> None:
        run_and_report(config, save_dashboard=False, print_summary=False)
        report_path = Path(config.output_dir) / "pipeline_report.txt"
        assert report_path.exists()
        content = report_path.read_text()
        assert "SPIKE ISLAND" in content


class TestGenerateTextReport:
    def test_contains_all_stages(self, config) -> None:
        result = run_pipeline(config)
        report = generate_text_report(result)
        for stage_name in ["SIMULATORS", "ANALYSIS", "WAVEFORMS", "SORTING",
                           "WILSON_COWAN", "STDP"]:
            assert stage_name.upper() in report.upper(), f"Missing {stage_name}"

    def test_contains_version(self, config) -> None:
        result = run_pipeline(config)
        report = generate_text_report(result)
        assert "Version:" in report


class TestPlotPipelineDashboard:
    def test_creates_png_file(self, config) -> None:
        result = run_pipeline(config)
        out_path = Path(config.output_dir) / "test_dashboard.png"
        saved = plot_pipeline_dashboard(result, output_path=out_path)
        assert saved.exists()
        assert saved.suffix == ".png"

    def test_file_has_reasonable_size(self, config) -> None:
        result = run_pipeline(config)
        out_path = Path(config.output_dir) / "test_dash2.png"
        plot_pipeline_dashboard(result, output_path=out_path)
        # Dashboard should be > 10 KB (non-empty image)
        assert out_path.stat().st_size > 10_000


# ===================================================================
# Integration: reproducibility test
# ===================================================================

class TestReproducibility:
    def test_same_seed_produces_same_spike_counts(self, config) -> None:
        r1 = run_pipeline(config)
        r2 = run_pipeline(config)
        s1 = r1.get_stage("simulators")
        s2 = r2.get_stage("simulators")
        assert s1 is not None and s2 is not None
        trains1 = s1.data["spike_trains"]
        trains2 = s2.data["spike_trains"]
        for name in trains1:
            assert len(trains1[name]) == len(trains2[name]), \
                f"{name} spike count differs across runs"
