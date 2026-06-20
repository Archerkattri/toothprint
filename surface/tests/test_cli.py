"""Tests for dentalmapcert.cli — covers all 89 statements."""
from __future__ import annotations

import pytest
from pathlib import Path

from dentalmapcert.cli import (
    _run_demo,
    _synthetic_lognormal_residuals,
    write_scaffold,
    main,
)


# ---------------------------------------------------------------------------
# _synthetic_lognormal_residuals
# ---------------------------------------------------------------------------

def test_synthetic_lognormal_returns_n_values():
    residuals = _synthetic_lognormal_residuals(n=10, mu=0.15, sigma=0.12, seed=42)
    assert len(residuals) == 10
    assert all(v >= 0.0 for v in residuals)


def test_synthetic_lognormal_is_sorted():
    residuals = _synthetic_lognormal_residuals(n=5, mu=0.15, sigma=0.12, seed=7)
    assert residuals == sorted(residuals)


def test_synthetic_lognormal_is_deterministic():
    a = _synthetic_lognormal_residuals(n=8, mu=0.2, sigma=0.1, seed=99)
    b = _synthetic_lognormal_residuals(n=8, mu=0.2, sigma=0.1, seed=99)
    assert a == b


def test_synthetic_lognormal_differs_with_different_seed():
    a = _synthetic_lognormal_residuals(n=5, mu=0.15, sigma=0.12, seed=1)
    b = _synthetic_lognormal_residuals(n=5, mu=0.15, sigma=0.12, seed=2)
    assert a != b


# ---------------------------------------------------------------------------
# write_scaffold
# ---------------------------------------------------------------------------

def test_write_scaffold_creates_three_files(tmp_path):
    paths = write_scaffold(tmp_path / "scaffold")
    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_write_scaffold_creates_output_dir_if_missing(tmp_path):
    target = tmp_path / "deep" / "scaffold"
    assert not target.exists()
    write_scaffold(target)
    assert target.is_dir()


# ---------------------------------------------------------------------------
# _run_demo
# ---------------------------------------------------------------------------

def test_run_demo_returns_two_certs(tmp_path):
    certs = _run_demo(tmp_path)
    assert len(certs) == 2


def test_run_demo_certs_have_labels(tmp_path):
    certs = _run_demo(tmp_path)
    for cert in certs:
        assert isinstance(cert.label, str) and len(cert.label) > 0


def test_run_demo_produces_the_two_certified_outcomes(tmp_path):
    # The demo must actually demonstrate what its docstring claims: one stable
    # and one change certificate (not two 'uncertain / recapture' rows).
    certs = _run_demo(tmp_path)
    assert certs[0].label == "surface stable certified"
    assert certs[1].label == "surface change certified"


def test_run_demo_certs_have_region_ids(tmp_path):
    certs = _run_demo(tmp_path)
    assert certs[0].surface_region_id == "case001_11_buccal"
    assert certs[1].surface_region_id == "case001_46_lingual"


def test_run_demo_writes_output_files(tmp_path):
    _run_demo(tmp_path)
    files = list(tmp_path.iterdir())
    assert len(files) >= 1


# ---------------------------------------------------------------------------
# main() — subcommand dispatch
# ---------------------------------------------------------------------------

def test_main_run_demo_returns_zero(tmp_path):
    rc = main(["run-demo", "--output-dir", str(tmp_path)])
    assert rc == 0


def test_main_write_scaffold_returns_zero(tmp_path):
    rc = main(["write-scaffold", "--output-dir", str(tmp_path)])
    assert rc == 0


def test_main_write_scaffold_default_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["write-scaffold"])
    assert rc == 0


def test_main_validate_dataset_returns_one_when_paths_missing(tmp_path):
    # Non-existent root → loader reports path errors → returns 1
    rc = main(["validate-dataset", "--dataset", "teeth3ds", "--root", str(tmp_path / "nonexistent")])
    assert rc == 1


def test_main_no_command_exits():
    with pytest.raises(SystemExit):
        main([])
