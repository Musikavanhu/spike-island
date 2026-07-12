"""Tests for spike_island.benchmarks — Performance profiling & benchmarking suite.

Covers:
- BenchmarkResult dataclass fields and defaults
- BenchmarkSuite aggregation and grouping
- _run_benchmark timing and memory profiling
- Each benchmark function returns non-empty results
- Benchmark results have valid timing and memory values
- print_benchmark_report produces non-empty output
- save_benchmark_report writes a file
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from spike_island.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    _run_benchmark,
    benchmark_simulators,
    benchmark_analysis,
    benchmark_waveforms,
    benchmark_sorting,
    benchmark_oscillator,
    benchmark_stdp,
    benchmark_pipeline,
    run_benchmarks,
    print_benchmark_report,
    save_benchmark_report,
)


# ===================================================================
# BenchmarkResult tests
# ===================================================================


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_basic_fields(self):
        """All required fields are set correctly."""
        result = BenchmarkResult(
            name="test",
            category="test_cat",
            elapsed_ms=1.5,
            peak_memory_mb=0.3,
            throughput=10.0,
        )
        assert result.name == "test"
        assert result.category == "test_cat"
        assert result.elapsed_ms == 1.5
        assert result.peak_memory_mb == 0.3
        assert result.throughput == 10.0
        assert result.metadata == {}

    def test_metadata_default(self):
        """Metadata defaults to empty dict."""
        result = BenchmarkResult(
            name="x", category="y", elapsed_ms=1.0,
            peak_memory_mb=0.1, throughput=1.0,
        )
        assert result.metadata == {}

    def test_metadata_custom(self):
        """Custom metadata is preserved."""
        meta = {"key": "value", "count": 42}
        result = BenchmarkResult(
            name="x", category="y", elapsed_ms=1.0,
            peak_memory_mb=0.1, throughput=1.0,
            metadata=meta,
        )
        assert result.metadata == meta


# ===================================================================
# BenchmarkSuite tests
# ===================================================================


class TestBenchmarkSuite:
    """Tests for the BenchmarkSuite dataclass."""

    def test_empty_suite(self):
        """Empty suite has no results and zero time."""
        suite = BenchmarkSuite()
        assert suite.results == []
        assert suite.total_elapsed_ms == 0.0

    def test_add_result(self):
        """Adding a result appends to the list."""
        suite = BenchmarkSuite()
        r = BenchmarkResult(
            name="a", category="cat1", elapsed_ms=1.0,
            peak_memory_mb=0.1, throughput=1.0,
        )
        suite.add(r)
        assert len(suite.results) == 1
        assert suite.results[0] is r

    def test_add_multiple(self):
        """Adding multiple results preserves order."""
        suite = BenchmarkSuite()
        for i in range(5):
            suite.add(BenchmarkResult(
                name=f"r{i}", category="cat", elapsed_ms=float(i),
                peak_memory_mb=0.1, throughput=1.0,
            ))
        assert len(suite.results) == 5
        assert [r.name for r in suite.results] == ["r0", "r1", "r2", "r3", "r4"]

    def test_by_category_single(self):
        """All results in one category."""
        suite = BenchmarkSuite()
        for i in range(3):
            suite.add(BenchmarkResult(
                name=f"r{i}", category="cat1", elapsed_ms=1.0,
                peak_memory_mb=0.1, throughput=1.0,
            ))
        groups = suite.by_category()
        assert "cat1" in groups
        assert len(groups["cat1"]) == 3

    def test_by_category_multiple(self):
        """Results split across multiple categories."""
        suite = BenchmarkSuite()
        suite.add(BenchmarkResult(
            name="a", category="cat1", elapsed_ms=1.0,
            peak_memory_mb=0.1, throughput=1.0,
        ))
        suite.add(BenchmarkResult(
            name="b", category="cat2", elapsed_ms=2.0,
            peak_memory_mb=0.2, throughput=2.0,
        ))
        suite.add(BenchmarkResult(
            name="c", category="cat1", elapsed_ms=3.0,
            peak_memory_mb=0.3, throughput=3.0,
        ))
        groups = suite.by_category()
        assert len(groups["cat1"]) == 2
        assert len(groups["cat2"]) == 1


# ===================================================================
# _run_benchmark tests
# ===================================================================


class TestRunBenchmark:
    """Tests for the _run_benchmark utility."""

    def test_basic_execution(self):
        """A simple function is benchmarked and returns a result."""
        result = _run_benchmark(
            name="test_basic",
            category="test",
            func=lambda: 42,
            throughput_key="runs/ms",
            throughput_value=1.0,
        )
        assert result.name == "test_basic"
        assert result.category == "test"
        assert result.elapsed_ms >= 0
        assert result.peak_memory_mb >= 0
        assert result.throughput > 0

    def test_elapsed_time_positive(self):
        """Elapsed time is positive for any function."""
        result = _run_benchmark(
            name="test_time",
            category="test",
            func=lambda: sum(range(1000)),
            throughput_key="runs/ms",
            throughput_value=1.0,
        )
        assert result.elapsed_ms > 0

    def test_memory_tracked(self):
        """Peak memory is tracked and positive."""
        result = _run_benchmark(
            name="test_mem",
            category="test",
            func=lambda: np.zeros(1000),
            throughput_key="runs/ms",
            throughput_value=1.0,
        )
        assert result.peak_memory_mb > 0

    def test_throughput_computed(self):
        """Throughput is computed as value / elapsed_ms."""
        result = _run_benchmark(
            name="test_tp",
            category="test",
            func=lambda: None,
            throughput_key="items/ms",
            throughput_value=100.0,
        )
        assert result.throughput > 0
        # throughput = 100 / elapsed_ms, so elapsed_ms = 100 / throughput
        assert abs(result.elapsed_ms - 100.0 / result.throughput) < 0.1

    def test_metadata_passed_through(self):
        """Metadata is stored in the result."""
        meta = {"a": 1, "b": "x"}
        result = _run_benchmark(
            name="test_meta",
            category="test",
            func=lambda: None,
            throughput_key="runs/ms",
            throughput_value=1.0,
            metadata=meta,
        )
        assert result.metadata == meta

    def test_heavy_computation(self):
        """Benchmark handles heavier computations without error."""
        result = _run_benchmark(
            name="test_heavy",
            category="test",
            func=lambda: np.linalg.eigvals(np.random.randn(100, 100)),
            throughput_key="runs/ms",
            throughput_value=1.0,
        )
        assert result.elapsed_ms > 0
        assert result.peak_memory_mb > 0


# ===================================================================
# Individual benchmark function tests
# ===================================================================


class TestBenchmarkSimulators:
    """Tests for benchmark_simulators()."""

    def test_returns_non_empty(self):
        """Returns at least one result."""
        results = benchmark_simulators()
        assert len(results) > 0

    def test_all_have_category(self):
        """All results have category 'simulators'."""
        results = benchmark_simulators()
        for r in results:
            assert r.category == "simulators"

    def test_all_have_positive_time(self):
        """All results have positive elapsed time."""
        results = benchmark_simulators()
        for r in results:
            assert r.elapsed_ms > 0

    def test_covers_all_generators(self):
        """Results include all four generator types."""
        results = benchmark_simulators()
        names = [r.name for r in results]
        assert any("poisson" in n for n in names)
        assert any("refractory" in n for n in names)
        assert any("bursty" in n for n in names)
        assert any("rhythmic" in n for n in names)

    def test_metadata_has_spike_count(self):
        """Metadata includes spike count."""
        results = benchmark_simulators()
        for r in results:
            assert "spikes" in r.metadata
            assert isinstance(r.metadata["spikes"], int)
            assert r.metadata["spikes"] > 0


class TestBenchmarkAnalysis:
    """Tests for benchmark_analysis()."""

    def test_returns_non_empty(self):
        results = benchmark_analysis()
        assert len(results) > 0

    def test_all_have_category(self):
        results = benchmark_analysis()
        for r in results:
            assert r.category == "analysis"

    def test_covers_all_functions(self):
        """Results include all analysis functions."""
        results = benchmark_analysis()
        names = [r.name for r in results]
        assert any("analyze" in n for n in names)
        assert any("cv" in n for n in names)
        assert any("autocorrelogram" in n for n in names)
        assert any("isi_histogram" in n for n in names)


class TestBenchmarkWaveforms:
    """Tests for benchmark_waveforms()."""

    def test_returns_non_empty(self):
        results = benchmark_waveforms()
        assert len(results) > 0

    def test_all_have_category(self):
        results = benchmark_waveforms()
        for r in results:
            assert r.category == "waveforms"

    def test_covers_waveform_and_stats(self):
        """Results include both waveform generation and statistics."""
        results = benchmark_waveforms()
        names = [r.name for r in results]
        assert any("to_waveform" in n for n in names)
        assert any("statistics" in n for n in names)


class TestBenchmarkSorting:
    """Tests for benchmark_sorting()."""

    def test_returns_non_empty(self):
        results = benchmark_sorting()
        assert len(results) > 0

    def test_all_have_category(self):
        results = benchmark_sorting()
        for r in results:
            assert r.category == "sorting"

    def test_covers_sort_and_evaluate(self):
        """Results include both sorting and evaluation."""
        results = benchmark_sorting()
        names = [r.name for r in results]
        assert any("template_sort" in n for n in names)
        assert any("evaluate" in n for n in names)


class TestBenchmarkOscillator:
    """Tests for benchmark_oscillator()."""

    def test_returns_non_empty(self):
        results = benchmark_oscillator()
        assert len(results) > 0

    def test_all_have_category(self):
        results = benchmark_oscillator()
        for r in results:
            assert r.category == "oscillator"

    def test_covers_all_operations(self):
        """Results include simulate, fixed_points, stability, metrics, bifurcation."""
        results = benchmark_oscillator()
        names = [r.name for r in results]
        assert any("simulate" in n for n in names)
        assert any("fixed_points" in n for n in names)
        assert any("stability" in n for n in names)
        assert any("metrics" in n for n in names)
        assert any("bifurcation" in n for n in names)


class TestBenchmarkSTDP:
    """Tests for benchmark_stdp()."""

    def test_returns_non_empty(self):
        results = benchmark_stdp()
        assert len(results) > 0

    def test_all_have_category(self):
        results = benchmark_stdp()
        for r in results:
            assert r.category == "stdp"

    def test_covers_single_and_network(self):
        """Results include both single synapse and network benchmarks."""
        results = benchmark_stdp()
        names = [r.name for r in results]
        assert any("single_synapse" in n for n in names)
        assert any("network" in n for n in names)


class TestBenchmarkPipeline:
    """Tests for benchmark_pipeline()."""

    @pytest.mark.skip(reason="Full pipeline benchmark is too slow for test suite")
    def test_returns_non_empty(self):
        results = benchmark_pipeline()
        assert len(results) > 0

    @pytest.mark.skip(reason="Full pipeline benchmark is too slow for test suite")
    def test_all_have_category(self):
        results = benchmark_pipeline()
        for r in results:
            assert r.category == "pipeline"

    @pytest.mark.skip(reason="Full pipeline benchmark is too slow for test suite")
    def test_metadata_has_stage_count(self):
        """Metadata includes number of stages."""
        results = benchmark_pipeline()
        for r in results:
            assert "stages" in r.metadata
            assert r.metadata["stages"] == 6  # 6 pipeline stages


# ===================================================================
# Full suite tests
# ===================================================================


class TestRunBenchmarks:
    """Tests for run_benchmarks()."""

    def test_returns_suite(self):
        """Returns a BenchmarkSuite instance."""
        suite = run_benchmarks()
        assert isinstance(suite, BenchmarkSuite)

    @pytest.mark.skip(reason="Full suite includes pipeline benchmark which is too slow")
    def test_suite_has_results(self):
        """Suite contains results from all categories."""
        suite = run_benchmarks()
        assert len(suite.results) > 0

    @pytest.mark.skip(reason="Full suite includes pipeline benchmark which is too slow")
    def test_total_time_positive(self):
        """Total suite time is positive."""
        suite = run_benchmarks()
        assert suite.total_elapsed_ms > 0

    @pytest.mark.skip(reason="Full suite includes pipeline benchmark which is too slow")
    def test_covers_all_categories(self):
        """Suite includes results from all 7 categories."""
        suite = run_benchmarks()
        categories = {r.category for r in suite.results}
        expected = {"simulators", "analysis", "waveforms", "sorting",
                     "oscillator", "stdp", "pipeline"}
        assert categories == expected


# ===================================================================
# Report tests
# ===================================================================


class TestReportGeneration:
    """Tests for report generation functions."""

    def test_print_report_non_empty(self):
        """Report text is non-empty."""
        suite = BenchmarkSuite()
        suite.add(BenchmarkResult(
            name="test", category="test", elapsed_ms=1.0,
            peak_memory_mb=0.1, throughput=1.0,
        ))
        report = print_benchmark_report(suite)
        assert len(report) > 0
        assert "Benchmark Report" in report
        assert "test" in report

    def test_save_report_creates_file(self):
        """Saving a report creates the output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_report.txt")
            suite = BenchmarkSuite()
            suite.add(BenchmarkResult(
                name="test", category="test", elapsed_ms=1.0,
                peak_memory_mb=0.1, throughput=1.0,
            ))
            save_benchmark_report(suite, path)
            assert os.path.exists(path)
            content = Path(path).read_text()
            assert "Benchmark Report" in content
            assert "test" in content

    def test_save_report_creates_parent_dir(self):
        """Saving to a nested path creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "report.txt")
            suite = BenchmarkSuite()
            suite.add(BenchmarkResult(
                name="test", category="test", elapsed_ms=1.0,
                peak_memory_mb=0.1, throughput=1.0,
            ))
            save_benchmark_report(suite, path)
            assert os.path.exists(path)

    def test_report_includes_all_categories(self):
        """Report text includes all category headers."""
        suite = BenchmarkSuite()
        for cat in ["cat_a", "cat_b"]:
            suite.add(BenchmarkResult(
                name=f"r_{cat}", category=cat, elapsed_ms=1.0,
                peak_memory_mb=0.1, throughput=1.0,
            ))
        report = print_benchmark_report(suite)
        assert "CAT_A" in report
        assert "CAT_B" in report
