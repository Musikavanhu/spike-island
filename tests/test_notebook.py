"""Tests for spike_island.notebook — Notebook generator module.

Verifies that:
- The generated notebook is valid nbformat v4
- It contains the expected number of code and markdown cells
- The cell sources reference all module functions
- Saving and loading round-trips correctly
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import nbformat
import pytest

from spike_island.notebook import (
    generate_and_save,
    generate_notebook,
    save_notebook,
    _code,
    _md,
)


@pytest.fixture()
def notebook() -> nbformat.notebooknode.NotebookNode:
    """Generate a fresh notebook for each test."""
    return generate_notebook()


@pytest.fixture()
def saved_path(tmp_path: Path) -> Path:
    """Generate and save notebook to a temp directory."""
    return generate_and_save(tmp_path / "test_walkthrough.ipynb")


class TestCellHelpers:
    """Test the low-level cell builder helpers."""

    def test_markdown_cell_type(self) -> None:
        cell = _md("Hello world")
        assert cell["cell_type"] == "markdown"
        assert "Hello world" in cell["source"]
        assert cell["metadata"] == {}
        assert cell["attachments"] == {}

    def test_code_cell_type(self) -> None:
        cell = _code("import numpy")
        assert cell["cell_type"] == "code"
        assert "import numpy" in cell["source"]
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        assert cell["metadata"] == {}

    def test_md_dedents_multiline(self) -> None:
        source = """
        ## Heading
        Some indented text.
        """
        cell = _md(source)
        # Leading blank line and indentation should be stripped
        assert "## Heading" in cell["source"]
        assert "Some indented text." in cell["source"]

    def test_code_dedents_multiline(self) -> None:
        source = """
        x = 1
        y = x + 1
        """
        cell = _code(source)
        assert "x = 1" in cell["source"]
        assert "y = x + 1" in cell["source"]


class TestNotebookStructure:
    """Verify the notebook has the expected structure and sections."""

    def test_is_v4(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        assert notebook.nbformat == 4
        assert notebook.nbformat_minor >= 0

    def test_has_kernelspec(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        assert "kernelspec" in notebook.metadata
        assert notebook.metadata["kernelspec"]["language"] == "python"

    def test_cell_count_reasonable(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        # We expect a substantive notebook: 20+ cells total
        assert len(notebook.cells) >= 20

    def test_has_markdown_cells(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        md_count = sum(1 for c in notebook.cells if c.cell_type == "markdown")
        assert md_count >= 10  # At least 10 markdown explanation cells

    def test_has_code_cells(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        code_count = sum(1 for c in notebook.cells if c.cell_type == "code")
        assert code_count >= 10  # At least 10 executable code cells

    def test_title_present(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        assert "# 🧠 Spike Island" in all_text or "Spike Island" in all_text

    def test_covers_simulators(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells)
        assert "simulators" in all_text.lower()
        assert "Poisson" in all_text

    def test_covers_analysis(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells)
        assert "analysis" in all_text.lower()
        assert "CV" in all_text or "coefficient of variation" in all_text

    def test_covers_waveforms(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells)
        assert "waveform" in all_text.lower()

    def test_covers_sorting(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells)
        assert "sorting" in all_text.lower()
        assert "template" in all_text.lower()

    def test_covers_wilson_cowan(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells)
        assert "wilson" in all_text.lower() and "cowan" in all_text.lower()

    def test_covers_stdp(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells)
        assert "stdp" in all_text.lower()
        assert "plasticity" in all_text.lower() or "synaptic" in all_text.lower()

    def test_covers_pipeline(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells)
        assert "pipeline" in all_text.lower()

    def test_covers_benchmarks(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells)
        assert "benchmark" in all_text.lower()


class TestNotebookCodeCells:
    """Verify code cells import from spike_island and use key functions."""

    def test_imports_spike_island(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        code_sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        all_code = "\n".join(code_sources)
        assert "from spike_island" in all_code

    def test_references_all_modules(
        self, notebook: nbformat.notebooknode.NotebookNode
    ) -> None:
        code_sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        all_code = "\n".join(code_sources)
        expected_modules = [
            "spike_island.simulators",
            "spike_island.analysis",
            "spike_island.waveforms",
            "spike_island.sorting",
            "spike_island.oscillator",
            "spike_island.stdp",
            "spike_island.pipeline",
            "spike_island.benchmarks",
        ]
        for mod in expected_modules:
            assert mod in all_code, f"Missing import from {mod}"

    def test_uses_poisson_spikes(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_code = "\n".join(
            c.source for c in notebook.cells if c.cell_type == "code"
        )
        assert "poisson_spikes" in all_code

    def test_uses_analyze(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_code = "\n".join(
            c.source for c in notebook.cells if c.cell_type == "code"
        )
        assert "analyze" in all_code

    def test_uses_generate_ap_template(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_code = "\n".join(
            c.source for c in notebook.cells if c.cell_type == "code"
        )
        assert "generate_ap_template" in all_code

    def test_uses_wilson_cowan_simulate(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_code = "\n".join(
            c.source for c in notebook.cells if c.cell_type == "code"
        )
        assert "simulate_wilson_cowan" in all_code

    def test_uses_stdp_run(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_code = "\n".join(
            c.source for c in notebook.cells if c.cell_type == "code"
        )
        assert "run_stdp" in all_code


class TestSaveLoad:
    """Verify that saving and loading notebooks works correctly."""

    def test_save_creates_file(self, saved_path: Path) -> None:
        assert saved_path.exists()
        assert saved_path.suffix == ".ipynb"

    def test_load_roundtrip(self, saved_path: Path) -> None:
        nb = nbformat.read(str(saved_path), as_version=4)
        assert nb.nbformat == 4
        assert len(nb.cells) > 0

    def test_json_valid(self, saved_path: Path) -> None:
        """The notebook file should be valid JSON (ipynb is JSON)."""
        content = saved_path.read_text()
        parsed = json.loads(content)
        assert "cells" in parsed
        assert "metadata" in parsed
        assert "nbformat" in parsed

    def test_save_notebook_explicit(
        self, tmp_path: Path, notebook: nbformat.notebooknode.NotebookNode
    ) -> None:
        out = tmp_path / "custom_name.ipynb"
        result = save_notebook(notebook, out)
        assert result == out.resolve()
        assert result.exists()

    def test_generate_and_save_roundtrip(self, tmp_path: Path) -> None:
        out = tmp_path / "auto.ipynb"
        path = generate_and_save(out)
        nb = nbformat.read(str(path), as_version=4)
        assert len(nb.cells) == len(generate_notebook().cells)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "deep.ipynb"
        path = generate_and_save(nested)
        assert path.exists()
        assert path == nested.resolve()


class TestNotebookContent:
    """Higher-level content checks on the notebook."""

    def test_has_installation_section(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        assert "pip install" in all_text

    def test_has_summary_table(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        # Should have a summary/next-steps section
        assert "Summary" in all_text or "summary" in all_text

    def test_has_next_steps(self, notebook: nbformat.notebooknode.NotebookNode) -> None:
        all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        assert "Next Steps" in all_text or "next steps" in all_text

    def test_cell_order_reasonable(
        self, notebook: nbformat.notebooknode.NotebookNode
    ) -> None:
        """Markdown title should come before code cells."""
        md_indices = [i for i, c in enumerate(notebook.cells) if c.cell_type == "markdown"]
        code_indices = [i for i, c in enumerate(notebook.cells) if c.cell_type == "code"]
        # First cell should be markdown (title)
        assert notebook.cells[0].cell_type == "markdown"
        # Both types should be present
        assert len(md_indices) > 0
        assert len(code_indices) > 0
