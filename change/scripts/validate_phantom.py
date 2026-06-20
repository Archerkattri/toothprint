#!/usr/bin/env python3
"""Validate the DCC system on phantom/typodont repeat-acquisition data.

All phantom pairs are stable (no real change). This script measures
the false-positive rate: fraction of phantom stable pairs incorrectly
certified as "progressed".

Usage:
    python scripts/validate_phantom.py --data data/phantom/extracted
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dcc.config import load_yaml
from dcc.data.phantom_adapter import PhantomAdapter
from dcc.score.periodontal import scalar_change_score


def main() -> None:
    _thresh = load_yaml("thresholds.yaml")
    tau_default = float(_thresh.get("bone_level_change", {}).get("tau_conservative_px", 10.0))

    parser = argparse.ArgumentParser(description="Phantom validation: measure false-positive rate")
    parser.add_argument("--data", default="data/phantom/extracted")
    parser.add_argument("--tau", type=float, default=tau_default)
    parser.add_argument("--output", default="outputs/phantom_validation")
    args = parser.parse_args()

    adapter = PhantomAdapter(args.data)
    pairs = adapter.stable_pairs()

    if not pairs:
        print(f"No phantom pairs found at {args.data}. Skipping validation.")
        print("To add phantom data: place DenPAR-style JSON annotations in:")
        print("  data/phantom/extracted/phantom_NNN/acq_000.json, acq_001.json, ...")
        sys.exit(0)

    fp = 0
    scores = []
    unscoreable = 0
    for pair in pairs:
        try:
            score = scalar_change_score(pair["baseline"], pair["followup"])
            scores.append(score)
            if score >= args.tau:
                fp += 1
        except (ValueError, KeyError):
            unscoreable += 1

    n = len(scores)
    if n == 0:
        print(
            f"ERROR: {len(pairs)} phantom pairs found but none were scoreable "
            f"({unscoreable} unscoreable — missing cej/crest_line?). Refusing to "
            f"report a false-positive rate over zero measurements.",
            file=sys.stderr,
        )
        sys.exit(1)
    fpr = fp / n
    print(f"Phantom validation: scored {n}/{len(pairs)} stable pairs ({unscoreable} unscoreable)")
    print(f"  False-positive rate (score >= tau={args.tau}): {fpr:.3f}")
    print(f"  Score mean: {sum(scores)/n:.3f}  max: {max(scores):.3f}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "n_pairs": n,
        "n_unscoreable": unscoreable,
        "false_positive_rate": fpr,
        "tau": args.tau,
    }
    (out_dir / "phantom_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results written to {out_dir}/phantom_validation.json")

if __name__ == "__main__":
    main()
