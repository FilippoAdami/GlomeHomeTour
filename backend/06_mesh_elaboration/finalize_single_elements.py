#!/usr/bin/env python3
"""Run Phase 06 Step 4: fit walls to final contours, then finalize ceilings."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from fit_enclosing_walls_to_regularized_contours import fit_enclosing_walls_to_regularized_contours
from regularize_ceiling_planes import regularize_ceiling_planes


def finalize_single_elements(workspace: Path) -> dict:
    wall = fit_enclosing_walls_to_regularized_contours(workspace)
    ceiling = regularize_ceiling_planes(workspace, wall_support=Path(wall["output_ply"]))
    return {"wall": wall, "ceiling": ceiling}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Scan workspace after Phase 06 Steps 1–3")
    args = parser.parse_args(argv)
    try:
        result = finalize_single_elements(args.workspace)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: Could not finalize Phase 06 single elements: {exc}", file=sys.stderr)
        return 1
    print(f"Step 4.1: fitted walls -> {result['wall']['output_ply']}")
    print(f"Step 4.2: regularized ceilings -> {result['ceiling']['output_ply']}")
    print(f"Ceiling manifest -> {result['ceiling']['manifest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
