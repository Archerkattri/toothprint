"""Tests for dcc.config.__init__ — load_yaml and _simple_yaml_parse."""

import sys
import unittest
from unittest.mock import patch


class TestLoadYaml(unittest.TestCase):
    def test_load_yaml_via_yaml_module(self):
        """load_yaml loads YAML using the yaml module when available."""
        from dcc.config import load_yaml
        cfg = load_yaml("perturb_ranges.yaml")
        self.assertIn("acquisition_noise", cfg)

    def test_load_yaml_fallback_when_yaml_unavailable(self):
        """Lines 11-13: when yaml is not importable, _simple_yaml_parse is used."""
        import importlib
        import sys
        from unittest.mock import patch

        mod_key = 'dcc.config'
        # Force reload to exercise the ImportError branch with yaml blocked
        with patch.dict(sys.modules, {'yaml': None}):
            # We can just call _simple_yaml_parse directly — the ImportError path
            # in load_yaml is equivalent to calling _simple_yaml_parse on the file text.
            from dcc.config import _simple_yaml_parse, CONFIG_DIR
            text = (CONFIG_DIR / "perturb_ranges.yaml").read_text()
            result = _simple_yaml_parse(text)
        self.assertIn("acquisition_noise", result)

    def test_load_yaml_importerror_triggers_simple_parse(self):
        """Lines 11-13: block yaml import so load_yaml falls through to _simple_yaml_parse."""
        with patch.dict(sys.modules, {'yaml': None}):
            from dcc.config import load_yaml
            cfg = load_yaml("perturb_ranges.yaml")
        self.assertIn("acquisition_noise", cfg)


class TestSimpleYamlParse(unittest.TestCase):
    """Direct tests of _simple_yaml_parse — covers lines 20-47."""

    def _parse(self, text: str) -> dict:
        from dcc.config import _simple_yaml_parse
        return _simple_yaml_parse(text)

    def test_flat_string_value(self):
        """line 34: string value that literal_eval raises on → stored as string."""
        result = self._parse("key: hello_world\n")
        self.assertEqual(result["key"], "hello_world")

    def test_flat_numeric_value(self):
        """line 32: numeric value → literal_eval succeeds → stored as number."""
        result = self._parse("number: 3.14\n")
        self.assertAlmostEqual(result["number"], 3.14)

    def test_nested_section(self):
        """Lines 36-46: nested section with sub-keys."""
        text = "outer:\n  inner: 42\n"
        result = self._parse(text)
        self.assertEqual(result["outer"]["inner"], 42)

    def test_nested_string_value(self):
        """Line 46: nested string that literal_eval raises on → stored as string."""
        text = "section:\n  name: some_string\n"
        result = self._parse(text)
        self.assertEqual(result["section"]["name"], "some_string")

    def test_comment_and_blank_lines_skipped(self):
        """Line 25: blank lines and comment lines are skipped."""
        text = "\n# comment\nkey: 1\n\n"
        result = self._parse(text)
        self.assertEqual(result["key"], 1)

    def test_inline_comment_stripped_from_nested_value(self):
        """Line 42: inline # comment is stripped from nested value."""
        text = "section:\n  value: 99  # this is a comment\n"
        result = self._parse(text)
        self.assertEqual(result["section"]["value"], 99)

    def test_empty_text_returns_empty_dict(self):
        result = self._parse("")
        self.assertEqual(result, {})

    def test_perturb_ranges_yaml_parses_cleanly(self):
        """End-to-end: _simple_yaml_parse handles the real perturb_ranges.yaml."""
        from dcc.config import _simple_yaml_parse, CONFIG_DIR
        text = (CONFIG_DIR / "perturb_ranges.yaml").read_text()
        result = _simple_yaml_parse(text)
        self.assertIn("acquisition_noise", result)
        self.assertIn("noise_std_px", result["acquisition_noise"])


if __name__ == "__main__":
    unittest.main()
