"""Tests for dentalmapcert.regions — 12 tests."""
from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dentalmapcert.regions import (
    TOOTH_SURFACES,
    all_region_ids,
    load_vertex_indices,
    region_id,
    region_surface_from_id,
    save_vertex_indices,
    tooth_arch,
    tooth_side,
    tooth_type,
)


# ---------------------------------------------------------------------------
# tooth_arch
# ---------------------------------------------------------------------------

class TestToothArch:
    def test_tooth_arch_upper(self):
        # Quadrant 1 (11-18) and quadrant 2 (21-28) → upper
        assert tooth_arch(16) == "upper"
        assert tooth_arch(21) == "upper"
        # Deciduous upper
        assert tooth_arch(53) == "upper"
        assert tooth_arch(62) == "upper"

    def test_tooth_arch_lower(self):
        # Quadrant 3 (31-38) and quadrant 4 (41-48) → lower
        assert tooth_arch(36) == "lower"
        assert tooth_arch(44) == "lower"
        # Deciduous lower
        assert tooth_arch(73) == "lower"
        assert tooth_arch(81) == "lower"

    def test_tooth_arch_invalid_raises(self):
        with pytest.raises(ValueError):
            tooth_arch(99)


# ---------------------------------------------------------------------------
# tooth_type
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tooth_side
# ---------------------------------------------------------------------------

class TestToothSide:
    def test_tooth_side_right(self):
        # Quadrant 1 (11-18) and quadrant 4 (41-48) → right
        assert tooth_side(14) == "right"
        assert tooth_side(41) == "right"

    def test_tooth_side_left(self):
        # Quadrant 2 (21-28) and quadrant 3 (31-38) → left
        assert tooth_side(23) == "left"
        assert tooth_side(36) == "left"

    def test_tooth_side_deciduous(self):
        # Deciduous quadrants 5/8 -> right, 6/7 -> left (mirrors tooth_arch).
        assert tooth_side(54) == "right"   # upper-right deciduous
        assert tooth_side(84) == "right"   # lower-right deciduous
        assert tooth_side(64) == "left"    # upper-left deciduous
        assert tooth_side(74) == "left"    # lower-left deciduous

    def test_tooth_side_invalid_raises(self):
        with pytest.raises(ValueError):
            tooth_side(99)

    def test_region_surface_round_trip_deciduous(self):
        # region_surface_from_id must invert region_id for deciduous FDI numbers.
        rid = region_id(54, "buccal")
        fdi, surface = region_surface_from_id(rid)
        assert fdi == 54 and surface == "buccal"


# ---------------------------------------------------------------------------
# tooth_type
# ---------------------------------------------------------------------------

class TestToothType:
    def test_tooth_type_molar(self):
        assert tooth_type(16) == "molar"
        assert tooth_type(27) == "molar"
        assert tooth_type(38) == "molar"

    def test_tooth_type_incisor(self):
        assert tooth_type(11) == "incisor"
        assert tooth_type(22) == "incisor"

    def test_tooth_type_canine(self):
        assert tooth_type(13) == "canine"
        assert tooth_type(43) == "canine"

    def test_tooth_type_premolar(self):
        # Tooth number 4 or 5 within quadrant → premolar
        assert tooth_type(14) == "premolar"
        assert tooth_type(25) == "premolar"

    def test_tooth_type_invalid_raises(self):
        # Tooth number 0 or 9 within quadrant is not valid
        with pytest.raises(ValueError):
            tooth_type(19)  # tooth_num=9 — no such tooth type


# ---------------------------------------------------------------------------
# region_id
# ---------------------------------------------------------------------------

class TestRegionId:
    def test_region_id_format(self):
        assert region_id(16, "buccal") == "tooth_16_buccal"
        assert region_id(21, "occlusal") == "tooth_21_occlusal"

    def test_region_id_invalid_surface_raises(self):
        with pytest.raises(ValueError, match="Unknown surface"):
            region_id(16, "vestibular")

    def test_all_region_ids_count(self):
        ids = all_region_ids(36)
        assert len(ids) == len(TOOTH_SURFACES)
        # Each should start with "tooth_36_"
        assert all(r.startswith("tooth_36_") for r in ids)


# ---------------------------------------------------------------------------
# region_surface_from_id (roundtrip)
# ---------------------------------------------------------------------------

class TestRegionSurfaceFromId:
    def test_region_surface_from_id_roundtrip(self):
        for fdi in (11, 16, 21, 36, 44):
            for surface in TOOTH_SURFACES:
                rid = region_id(fdi, surface)
                fdi_out, surface_out = region_surface_from_id(rid)
                assert fdi_out == fdi
                assert surface_out == surface

    def test_region_surface_from_id_malformed_raises(self):
        with pytest.raises(ValueError):
            region_surface_from_id("bad_id")

    def test_region_surface_from_id_bad_surface_raises(self):
        with pytest.raises(ValueError):
            region_surface_from_id("tooth_16_vestibular")

    def test_region_surface_from_id_non_integer_fdi_raises(self):
        with pytest.raises(ValueError, match="Non-integer FDI"):
            region_surface_from_id("tooth_abc_buccal")


# ---------------------------------------------------------------------------
# load / save vertex indices
# ---------------------------------------------------------------------------

class TestVertexIndices:
    def test_load_vertex_indices_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_vertex_indices("/tmp/does_not_exist_xyz_123.npy")

    def test_save_and_load_vertex_indices_roundtrip(self):
        indices = np.array([0, 5, 10, 100, 999], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "region_indices.npy"
            out_path = save_vertex_indices(indices, path)
            assert out_path == path
            loaded = load_vertex_indices(path)
            np.testing.assert_array_equal(loaded, indices)
            assert loaded.dtype == np.int64
