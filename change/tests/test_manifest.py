import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ManifestTests(unittest.TestCase):
    def test_default_manifest_records_denpar_as_open_core_dataset(self):
        from dcc.data.manifest import build_default_manifest

        manifest = build_default_manifest()

        denpar = manifest.get_source("denpar")
        self.assertEqual(denpar.license, "CC-BY-4.0")
        self.assertTrue(denpar.redistribution_allowed)
        self.assertEqual(denpar.access, "public")
        self.assertEqual(denpar.role, "core")
        self.assertIn("intraoral periapical", denpar.modality.lower())
        self.assertIn("CEJ keypoints", denpar.annotation_types)
        self.assertIn("alveolar crest lines", denpar.annotation_types)

    def test_default_manifest_quarantines_noncommercial_caries_source(self):
        from dcc.data.manifest import build_default_manifest

        manifest = build_default_manifest()

        mendeley = manifest.get_source("mendeley_bitewing_caries")
        self.assertEqual(mendeley.license, "CC-BY-NC-3.0")
        self.assertFalse(mendeley.redistribution_allowed)
        self.assertEqual(mendeley.role, "optional-nc")
        self.assertIn("excluded from open release", mendeley.limitations.lower())

    def test_default_manifest_records_periapical_lesion_shift_source(self):
        from dcc.data.manifest import build_default_manifest

        manifest = build_default_manifest()

        periapical = manifest.get_source("periapical_lesions_radiographs")
        self.assertEqual(periapical.license, "CC-BY-4.0")
        self.assertTrue(periapical.redistribution_allowed)
        self.assertEqual(periapical.access, "public")
        self.assertEqual(periapical.role, "external-shift")
        self.assertIn("periapical", periapical.modality.lower())
        self.assertIn("periapical lesion binary labels", periapical.annotation_types)

    def test_manifest_writer_emits_json_with_license_flags(self):
        from dcc.data.manifest import build_default_manifest, write_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = write_manifest(build_default_manifest(), Path(tmpdir) / "dataset_manifest.json")
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "0.1")
        ids = {source["id"] for source in payload["sources"]}
        self.assertIn("denpar", ids)
        self.assertIn("mendeley_bitewing_caries", ids)
        self.assertIn("periapical_lesions_radiographs", ids)

    def test_get_source_unknown_id_raises_key_error(self):
        """DatasetManifest.get_source raises KeyError for an unknown source_id (line 39)."""
        from dcc.data.manifest import build_default_manifest
        manifest = build_default_manifest()
        with self.assertRaises(KeyError):
            manifest.get_source("nonexistent_source_xyz")

    def test_source_ids_returns_set_of_all_source_ids(self):
        """source_ids() returns a set of all source ids (line 126)."""
        from dcc.data.manifest import build_default_manifest, source_ids
        manifest = build_default_manifest()
        ids = source_ids(manifest.sources)
        self.assertIsInstance(ids, set)
        self.assertIn("denpar", ids)
        self.assertIn("mendeley_bitewing_caries", ids)
        self.assertIn("periapical_lesions_radiographs", ids)

    def test_main_function_writes_manifest_and_returns_zero(self):
        """main() writes the manifest JSON and returns 0 (lines 130-140)."""
        from dcc.data.manifest import main
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "manifest.json"
            result = main(["--output", str(out)])
            self.assertEqual(result, 0)
            self.assertTrue(out.exists())

    def test_manifest_module_emits_output_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset_manifest.json"
            env = dict(os.environ)
            env.update({"CUDA_VISIBLE_DEVICES": "0"})

            subprocess.run(
                [sys.executable, "-m", "dcc.data.manifest", "--output", str(output_path)],
                check=True,
                env=env,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "0.1")
        self.assertEqual(payload["sources"][0]["id"], "denpar")


if __name__ == "__main__":
    unittest.main()
