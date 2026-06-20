"""Clinical configuration defaults."""
from pathlib import Path

CONFIG_DIR = Path(__file__).parent


def load_yaml(name: str) -> dict:
    """Load a YAML config file from dcc/config/. Returns dict."""
    try:
        import yaml
    except ImportError:
        # Fallback to manual parse for the simple scalars we use
        return _simple_yaml_parse((CONFIG_DIR / name).read_text())
    with (CONFIG_DIR / name).open() as f:
        return yaml.safe_load(f)


def _simple_yaml_parse(text: str) -> dict:
    """Very minimal YAML parser for flat scalar/list configs (no PyYAML needed)."""
    import ast
    result = {}
    current_key = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if not line.startswith(' ') and ':' in stripped:
            key, _, val = stripped.partition(':')
            val = val.strip()
            if val:
                try:
                    result[key.strip()] = ast.literal_eval(val)
                except Exception:
                    result[key.strip()] = val
            else:
                current_key = key.strip()
                result[current_key] = {}
        elif current_key and ':' in stripped:
            key, _, val = stripped.partition(':')
            val = val.strip()
            # strip inline comment
            val = val.split('#')[0].strip()
            try:
                result[current_key][key.strip()] = ast.literal_eval(val)
            except Exception:
                result[current_key][key.strip()] = val
    return result
